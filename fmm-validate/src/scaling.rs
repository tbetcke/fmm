//! Strong and weak scaling of the distributed host FMM (Phase 5 T10, C5.3): the core of
//! the `scaling` example.
//!
//! One [`run`] builds one `nd_fmm_exec::fmm::Fmm` on every rank of a communicator,
//! evaluates it repeatedly and reports, per rank and reduced over the ranks
//! ([`Report`]):
//! - the work as built: input points, owned sources and targets, leaves, list pairs, held
//!   boxes (own, halo, shared ancestors, other top-tree keys of the locally essential
//!   set, top-tree keys held only for the global pass; Phase 5S T4);
//! - the build by part (`BuildTimings`: domain, octree, plan, redistribution, tables,
//!   evaluator with its exchange builds, load), the median over [`Settings::builds`];
//! - the evaluation by stage ([`STAGES`]): the median per rank over
//!   [`Settings::evaluations`] after one warm-up, then the maximum, minimum and mean over
//!   the ranks and the load imbalance (maximum over mean) of the compute stages;
//! - per exchange the messages and bytes per rank ([`RankTraffic`]);
//! - memory per rank: the formula of docs/design/distributed-fmm.md §9.2, as
//!   docs/design/scale-out.md §8.2 updates it (Phase 5S T6), evaluated ([`MemoryModel`], a
//!   *model*), and the resident size where the machine reports it
//!   without a new dependency ([`resident_memory`]);
//! - optionally the errors against the direct sum at sampled targets ([`Errors`]) and the
//!   difference from the one-rank `Fmm` over every point in index order ([`Reference`]),
//!   of the strategy the run resolved or of another ([`Settings::reference_strategy`],
//!   Phase 5N T3);
//! - the M2L strategy as resolved and why (Phase 5N T4): under `Auto` the rule of
//!   `M2lStrategy::resolve_on_node` with the agreed ranks per node and the node budget
//!   ([`Settings::dense_node_budget`]; the harness sets it per machine class), in the
//!   threading line of the report and as `requested`, `budget_mb`, `dense_rank_mb` and
//!   `dense_node_mb` in the summary;
//! - optionally the time of every level call by operator kind ([`Settings::kinds`], Phase
//!   5N T3): per rank and kind the median over the evaluations ([`KINDS`]), and the M2L
//!   and downward time per V pair of the slowest rank;
//! - where the ranks ran ([`Placement`], Phase 5N T2): the nodes and the ranks per node
//!   (from `split_shared`), each rank's host and CPUs, the MPI library, and on a cluster
//!   the Slurm job with its QoS, node list and binding ([`SlurmJob`]);
//! - the harness's own memory per rank (Phase 5S T2): the resident size after the input is
//!   generated, before the builds ([`RankRow::resident_input`]);
//! - in the overlapped path, from `OverlapTimes` (Phase 5S T9, "The exposed collectives"):
//!   per rank the median of the `test` calls of each exchange, the multipole exchange's
//!   final wait for its sends and its wait for each level ([`PathTimes::overlap`],
//!   [`OVERLAP_PARTS`]), reduced over the ranks in a table of the report and as
//!   `multipoles_progress_max_ms`, `multipoles_sends_wait_max_ms`, `level_wait_max_ms` and
//!   `level_wait_max_level` in the overlapped path's summary.
//!
//! [`invariance`] (Phase 5S T2, C5S.1) is the other oracle at scale: the run on P ranks
//! against the run on P/k ranks, bit for bit ("Rank-count invariance" below).
//!
//! Nothing here asserts a timing; the smoke test checks the fields and the errors.
//!
//! # The workload
//!
//! A [`Workload`] generates **every point by its index** (docs/design/distributed-fmm.md
//! §14.4, S7): point i is drawn from its own generator, seeded by a hash of the seed and
//! i, so a rank produces its own points without drawing the others', and the problem of
//! N points is the first N points of every larger one (the weak-scaling problems are
//! nested). The charges of vector k are drawn the same way from `seed + 1 + k`. The
//! distributions are those of [`fmm_accuracy`](crate::fmm_accuracy) (design §8.2), drawn
//! point by point; the Gaussian clusters take point i in cluster i mod 5. The tree is the
//! adaptive one of the C3.3 problems: `max_level` 16 and 64 points per leaf.
//!
//! # Input distributions
//!
//! [`Input`]: every point on rank 0; a seeded random share (point i on rank
//! `hash(i) mod P`); contiguous index ranges (Phase 5S T2: found in O(N/P) time, where the
//! share hashes every index on every rank, O(N); docs/design/scale-out.md §5.2 keeps the
//! share up to N = 10⁹); or every point already on the rank that owns its leaf, found by a
//! first build from the share (the partition does not depend on the input, Phase 5 T4).
//! Every input but the owners' is found from the indices alone ([`input_by_index`]), so any
//! rank can produce any rank's input, and no rank holds O(N) (but rank 0 with every point).
//! In every case the output is that of the one-rank `Fmm` over the union of the inputs in
//! rank order, bit for bit, and within 100 u_T of the one-rank `Fmm` over the points in
//! index order (docs/design/distributed-fmm.md §5), which [`Reference`] measures.
//!
//! # Errors against the direct sum
//!
//! As in [`fmm_accuracy`](crate::fmm_accuracy): the root mean squares over the charge
//! vectors of the relative L2 and max errors of φ and ∇φ at the sampled targets, against
//! `direct_sum` in f64 over every source with the charges rounded to the FMM's precision,
//! divided by 4π. The direct sum is computed once over the ranks, each rank over the
//! sources it passed (`direct_sum`: in f64, Neumaier-compensated over the rank's sources),
//! and the ranks' parts added by one all-reduce, so it changes with the rank count only
//! in the last bits. Its cost is O(N/P) per rank, the rank's sources times the sampled
//! targets ([`Errors::direct_pairs`], timed in [`Errors::direct_seconds`]), and nothing in
//! it holds O(N): it is the oracle at N ≥ 10⁸ (Phase 5S T2), where no one-rank reference
//! fits ([`REFERENCE_MAX_POINTS`]).
//!
//! # The one-rank reference
//!
//! [`Settings::reference`] gathers every rank's output and indices to rank 0, which builds
//! the one-rank `Fmm` over every point in index order. Above [`REFERENCE_MAX_POINTS`]
//! (2 × 10⁷) [`run`] refuses it with [`HarnessError::ReferenceTooLarge`], on every rank
//! before any collective.
//!
//! # Rank-count invariance
//!
//! [`invariance`] (requirement 3 of docs/phase5s/README.md; docs/design/scale-out.md §5.1;
//! S7 of docs/design/distributed-fmm.md §14.4): a run on P ranks equals a run on P/k ranks
//! bit for bit when rank r of the small run holds the concatenation of the inputs of ranks
//! k r, …, k r + k − 1 of the large one, in that order. Both runs are in one launch: the
//! small one on a split of the communicator, with the strategy of the large one named
//! (`Auto` resolves by the ranks per node, which differ); the outputs move back by index,
//! point to point, and are never gathered. The test of `nd-fmm-exec`
//! (`tests/invariance.rs`) checks the same at N = 10⁴ on 1, 2 and 4 ranks in CI; this is
//! the check at scale (P against P/8: 512 against 64 on locust, 160 against 20 and 640
//! against 80 on Kathleen).

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::time::{Duration, Instant};

use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::{Communicator, CommunicatorCollectives, Destination, Equivalence, Root, Source};
use nd_fmm_exec::cost::CostModel;
use nd_fmm_exec::fmm::{
    Backend, BuildTimings, DeviceGather, DeviceGemm, DeviceReduction, ExchangeTraffic, Fmm,
    FmmBuilder, KindTiming, ListSizes, MultipoleRouting, NearRuleChoice, OperatorKind, Output,
    OutputMany, StageTimings, TREE_FAN_OUT, Traffic,
};
use nd_fmm_exec::geometry::{contains, integer_centre};
use nd_fmm_exec::operator::{MulAdd, P2pChoice, SimdScalar};
use nd_fmm_exec::tables::{M2lStrategy, NodeContext};
use nd_fmm_exec::threading::ThreadingReport;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;

use crate::bench::{cores, cpu_model, target, toolchain};
use crate::fmm_accuracy::{CLUSTER_CENTRES, CLUSTER_WIDTH, Distribution, PLUMMER_SCALE};
use crate::metrics::{ErrorAccumulator, ErrorNorms};
use crate::{SplitMix64, points};

/// The deepest level of a leaf: that of the C3.3 problems.
pub const MAX_LEVEL: usize = 16;

/// The refinement target: 64 points per leaf, the host path's default.
pub const MAX_POINTS_PER_LEAF: usize = 64;

/// Values per target in the output: φ and ∇φ.
const OUTPUT_VALUES: usize = 4;

/// The largest N of [`Settings::reference`], 2 × 10⁷: the one-rank reference gathers every
/// point and output to rank 0 and builds the one-rank `Fmm` there, and this is the largest
/// N that ran on one rank in Phase 5N (docs/design/scale-out.md §5.2). Above it the
/// oracles are the errors against the direct sum and rank-count invariance ([`invariance`]).
pub const REFERENCE_MAX_POINTS: usize = 20_000_000;

/// Why [`run`] or [`invariance`] refuses its arguments (Phase 5S T2). Each is decided from
/// the arguments alone, which every rank passes the same (the functions are collective), so
/// every rank returns the same error before any collective: an agreed error without a
/// collective of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HarnessError {
    /// [`Settings::reference`] with more than [`REFERENCE_MAX_POINTS`] points.
    ReferenceTooLarge {
        /// The number of points.
        n: usize,
        /// [`REFERENCE_MAX_POINTS`].
        max: usize,
    },
    /// A factor k of [`invariance`] that is 0 or does not divide the ranks.
    Factor {
        /// The factor.
        k: usize,
        /// The ranks of the communicator.
        ranks: usize,
    },
    /// [`invariance`] with an input that is not found from the indices alone
    /// ([`Input::Owners`]).
    InputNotByIndex(Input),
    /// [`Settings::rhs`] above 1 with a check that evaluates one vector: the errors
    /// against the direct sum, the one-rank reference or [`invariance`] (Phase 6 T5).
    SeveralVectors(&'static str),
}

impl fmt::Display for HarnessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceTooLarge { n, max } => write!(
                f,
                "the one-rank reference gathers every point to rank 0 and builds the FMM \
                 there; N = {n} is above {max}, the largest it runs at (docs/design/\
                 scale-out.md §5.2): check the errors against the direct sum (--errors) and \
                 rank-count invariance (--invariance) instead"
            ),
            Self::Factor { k, ranks } => write!(
                f,
                "rank-count invariance at k = {k}: k must divide the {ranks} ranks (the small \
                 run has P / k ranks)"
            ),
            Self::InputNotByIndex(input) => write!(
                f,
                "rank-count invariance needs an input found from the indices alone (share, \
                 ranges or rank0), not {}",
                input.name()
            ),
            Self::SeveralVectors(what) => write!(
                f,
                "{what} evaluates one charge vector; run it without --rhs (each vector of \
                 several is bit for bit its single-vector evaluation, Phase 6 T5)"
            ),
        }
    }
}

impl std::error::Error for HarnessError {}

/// A problem whose points and charges are generated by index (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workload {
    /// The point distribution.
    pub distribution: Distribution,
    /// The number of points, sources equal to targets.
    pub n: usize,
    /// The seed of the points, the share and the sample; charge vector k uses
    /// `seed + 1 + k`.
    pub seed: u64,
}

/// The seed of the workloads of the `scaling` example.
pub const SEED: u64 = 0x5ca1e;

/// What [`hash`] mixes into the seed for the random share and for the sample, so that
/// neither correlates with the points.
const SHARE_SALT: u64 = 0x5_4a7e;
const SAMPLE_SALT: u64 = 0x5_a3b1e;

/// A hash of `seed` and `i`: SplitMix64's output on the mixed seed XOR i, so that
/// successive indices give unrelated generators.
pub fn hash(seed: u64, i: u64) -> u64 {
    let mixed = SplitMix64::new(seed).next_u64();
    SplitMix64::new(mixed ^ i).next_u64()
}

impl Workload {
    /// N points of `distribution` from [`SEED`].
    pub const fn new(distribution: Distribution, n: usize) -> Self {
        Self {
            distribution,
            n,
            seed: SEED,
        }
    }

    /// Point `i`, drawn from its own generator (module documentation).
    pub fn point(&self, i: usize) -> [f64; 3] {
        let mut rng = SplitMix64::new(hash(self.seed, i as u64));
        let origin = [0.0; 3];
        let drawn = match self.distribution {
            Distribution::Cube => points::cube(&mut rng, 1, origin, 1.0),
            Distribution::Sphere => points::sphere(&mut rng, 1, origin, 1.0),
            Distribution::Plummer => points::plummer(&mut rng, 1, origin, PLUMMER_SCALE),
            Distribution::Clusters => points::gaussian_clusters(
                &mut rng,
                1,
                &CLUSTER_CENTRES[i % CLUSTER_CENTRES.len()..][..1],
                CLUSTER_WIDTH,
            ),
        };
        drawn[0]
    }

    /// The charge of point `i` in charge vector `vector`, uniform in [−1, 1).
    pub fn charge(&self, vector: usize, i: usize) -> f64 {
        let seed = self.seed.wrapping_add(1 + vector as u64);
        points::charges(&mut SplitMix64::new(hash(seed, i as u64)), 1)[0]
    }

    /// The rank of `size` that point `i` goes to in the random share.
    pub fn share(&self, i: usize, size: usize) -> usize {
        (hash(self.seed ^ SHARE_SALT, i as u64) % size.max(1) as u64) as usize
    }

    /// The indices of rank `rank` of `size` in [`Input::Ranges`]: from ⌊rank N / size⌋ to
    /// ⌊(rank + 1) N / size⌋.
    pub fn range(&self, rank: usize, size: usize) -> std::ops::Range<usize> {
        let bound = |r: usize| (r as u128 * self.n as u128 / size.max(1) as u128) as usize;
        bound(rank)..bound(rank + 1)
    }

    /// `count` distinct indices of targets, seeded, in the order drawn.
    ///
    /// # Panics
    ///
    /// If `count > n`.
    pub fn sample(&self, count: usize) -> Vec<usize> {
        assert!(count <= self.n, "more samples than points");
        let mut rng = SplitMix64::new(hash(self.seed ^ SAMPLE_SALT, 0));
        let mut seen = BTreeSet::new();
        let mut sample = Vec::with_capacity(count);
        while sample.len() < count {
            let i = (((rng.next_u64() >> 32) * self.n as u64) >> 32) as usize;
            if seen.insert(i) {
                sample.push(i);
            }
        }
        sample
    }
}

/// Where the points are when `build` is called (docs/phase5/README.md, "Workloads").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// Every point on rank 0.
    Rank0,
    /// Point i on rank `hash(i) mod P` ([`Workload::share`]).
    Share,
    /// Every point on the rank that owns its leaf.
    Owners,
    /// Contiguous index ranges, rank r the indices from ⌊r N / P⌋ to ⌊(r + 1) N / P⌋
    /// ([`Workload::range`]; Phase 5S T2): found in O(N/P) time, where the share filters
    /// every index on every rank (O(N); docs/design/scale-out.md §5.2: the share up to
    /// N = 10⁹, ranges beyond).
    Ranges,
}

impl Input {
    /// Every input distribution.
    pub const ALL: [Self; 4] = [Self::Rank0, Self::Share, Self::Owners, Self::Ranges];

    /// The name: "rank0", "share", "owners" or "ranges".
    pub fn name(self) -> &'static str {
        match self {
            Self::Rank0 => "rank0",
            Self::Share => "share",
            Self::Owners => "owners",
            Self::Ranges => "ranges",
        }
    }

    /// The input of a [`name`](Self::name), if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.name() == name)
    }

    /// A description, for reports.
    pub fn description(self) -> &'static str {
        match self {
            Self::Rank0 => "every point on rank 0",
            Self::Share => "a seeded random share (point i on rank hash(i) mod P)",
            Self::Owners => "every point on the rank that owns its leaf",
            Self::Ranges => {
                "contiguous index ranges (rank r the indices from r N / P to (r + 1) N / P)"
            }
        }
    }
}

/// Which evaluation paths a [`run`] times.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Overlap {
    /// Blocking exchanges only.
    Off,
    /// Overlapped exchanges only (`FmmBuilder::overlap`, Phase 5 T9).
    On,
    /// Both on one build, the evaluations alternating (`Fmm::set_overlap`), a block each.
    Both,
}

impl Overlap {
    /// The name: "off", "on" or "both".
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
            Self::Both => "both",
        }
    }

    /// The overlap of a [`name`](Self::name), if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Off, Self::On, Self::Both]
            .into_iter()
            .find(|o| o.name() == name)
    }

    /// The paths, blocking first.
    fn paths(self) -> &'static [bool] {
        match self {
            Self::Off => &[false],
            Self::On => &[true],
            Self::Both => &[false, true],
        }
    }
}

/// How a [`run`] builds, evaluates and checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// The degree; gradients are always on.
    pub p: usize,
    /// Rayon threads per rank (`FmmBuilder::threads`).
    pub threads: usize,
    /// The M2L strategy (`FmmBuilder::strategy`; the default `Auto`).
    pub strategy: M2lStrategy,
    /// The node budget of `Auto` in bytes (`FmmBuilder::dense_node_budget`; Phase 5N T4),
    /// or the library's default (`M2lStrategy::AUTO_DENSE_NODE_BYTES`, 64 MB) if `None`.
    pub dense_node_budget: Option<usize>,
    /// The multiply–add mode of the batched M2L (`FmmBuilder::m2l_mul_add`, Phase 6 T2;
    /// the library's `Separate`): only `M2lStrategy::Batched` uses it.
    pub m2l_mul_add: MulAdd,
    /// The evaluation paths.
    pub overlap: Overlap,
    /// Where the points are passed in.
    pub input: Input,
    /// Builds; the build times are the median over them, and the last is evaluated.
    pub builds: usize,
    /// Timed evaluations per path, after one warm-up each.
    pub evaluations: usize,
    /// Charge vectors of the errors against the direct sum; none if 0.
    pub vectors: usize,
    /// Sampled targets of the errors.
    pub sampled: usize,
    /// Whether to compare with the one-rank `Fmm` over every point in index order, built
    /// on rank 0 alone ([`Reference`]); refused above [`REFERENCE_MAX_POINTS`].
    pub reference: bool,
    /// The strategy of that one-rank `Fmm` (Phase 5N T3): the difference between two
    /// strategies at otherwise fixed settings. Without it, the strategy the run resolved
    /// (`Fmm::strategy`), whose output the run must equal bit for bit with the points in
    /// rank order; under `Auto` a one-rank `Fmm` could resolve another (Phase 5N T4).
    pub reference_strategy: Option<M2lStrategy>,
    /// Whether every level call is timed by operator kind (`KindTiming::Synchronous`, one
    /// clock pair per call on the host; Phase 5N T3), into [`PathTimes::kinds`].
    pub kinds: bool,
    /// Whether the blocks are cut a second time, by modelled cost
    /// (`FmmBuilder::work_cut`, Phase 5S T7; on, as the library's default).
    pub work_cut: bool,
    /// The constants of the modelled cost (`FmmBuilder::cost_model`), or the library's
    /// (`CostModel::calibrated`) if `None`; they weigh the second cut and give
    /// [`RankRow::modelled`].
    pub cost_model: Option<CostModel>,
    /// How the multipole exchange routes the multipoles (`FmmBuilder::multipole_routing`,
    /// Phase 5S T8s; the forwarding tree with `TREE_FAN_OUT`, as the library's default).
    pub routing: MultipoleRouting,
    /// The charge vectors of every timed evaluation (`FmmBuilder::vectors`, Phase 6 T5;
    /// 1, `Fmm::evaluate`): above 1 the FMM is built for that many and evaluated by
    /// `Fmm::evaluate_many`, vector k with the charges of seed + 1 + k; the errors, the
    /// reference and [`invariance`] need 1 ([`HarnessError::SeveralVectors`]).
    pub rhs: usize,
    /// Where the operators run (`FmmBuilder::backend`; the host by default): a device
    /// backend of the `gpu` features (Phase 6 T5, for the device's throughput per charge
    /// vector), on one rank or on the ranks sharing a device (correctness, never scaling).
    pub backend: Backend,
    /// The P2P kernel (`FmmBuilder::p2p_kernel`; the library's `Auto`, the ISA of
    /// `Isa::detect`): an explicit ISA, such as AVX-512 (Phase 6 T6), or the reference.
    /// On the host it also sets the ISA of the batched M2L's kernel.
    pub p2p: P2pChoice,
    /// The device's chunk budget in bytes (`FmmBuilder::device_scratch_budget`, Phase 6
    /// T7); `None` for the library's, per charge vector.
    pub device_scratch: Option<u64>,
    /// The device's GEMM of the grouped translations (`FmmBuilder::device_gemm`, Phase 6
    /// T7).
    pub device_gemm: DeviceGemm,
    /// How the device's hand-written GEMM gets its columns (`FmmBuilder::device_gather`,
    /// Phase 6 T7).
    pub device_gather: DeviceGather,
    /// The device's reduction (`FmmBuilder::device_reduction`, Phase 6 T7).
    pub device_reduction: DeviceReduction,
    /// The near rule (`FmmBuilder::near_rule`, Phase 6 T12, C6.9; off, as the library's
    /// default): the W and X pairs between leaves below its crossovers run as P2P.
    pub near_rule: NearRuleChoice,
}

impl Settings {
    /// Degree `p` on one thread, the default strategy, overlap off, the random share, one
    /// build, ten evaluations, no checks, no kind timings, the second cut on with the
    /// library's constants, the multipoles along the default forwarding tree.
    pub const fn new(p: usize) -> Self {
        Self {
            p,
            threads: 1,
            strategy: M2lStrategy::Auto,
            dense_node_budget: None,
            m2l_mul_add: MulAdd::Separate,
            overlap: Overlap::Off,
            input: Input::Share,
            builds: 1,
            evaluations: 10,
            vectors: 0,
            sampled: 1000,
            reference: false,
            reference_strategy: None,
            kinds: false,
            work_cut: true,
            cost_model: None,
            routing: MultipoleRouting::Tree {
                fan_out: TREE_FAN_OUT,
            },
            rhs: 1,
            backend: Backend::Host,
            p2p: P2pChoice::Auto,
            device_scratch: None,
            device_gemm: DeviceGemm::Auto,
            device_gather: DeviceGather::Auto,
            device_reduction: DeviceReduction::ByChunk,
            near_rule: NearRuleChoice::Off,
        }
    }
}

/// The stages of an evaluation as a [`run`] reports them, from `StageTimings`; times in
/// seconds, per rank the median over the evaluations.
///
/// The exchanges have three rows each: the stage of `StageTimings` (in the overlapped
/// path the rest of the overlapped stage outside the level calls), the exchange from post
/// to completion, and its exposed wait. In the blocking path the last two are the
/// stage itself: a blocking exchange is all exposed. "upward global" is the upward
/// reduction with the global M2M of the owned `Global` boxes and the all-gather of the
/// replicated top (Phase 5S T5; the coarse gather and the global M2M of every `Global` box
/// before); its parts "upward reduction" and "replicated top" (`StageTimings::reduction`
/// and `replicated_top`, Phase 5S T6) are timed in both paths, and are not added to the
/// total. Until T6 the harness had a row "top all-gather" in their place, timed only in
/// the overlapped path (`OverlapTimes::top_gather`, still the all-gather part of
/// "replicated top" there).
pub const STAGES: [&str; NSTAGES] = [
    "forward charges",
    "load",
    "source exchange (stage)",
    "sources: post to completion",
    "sources: exposed wait",
    "upward local",
    "upward global (reduction + global M2M + top)",
    "upward reduction (part of upward global)",
    "replicated top (part of upward global)",
    "multipole exchange (stage)",
    "multipoles: post to completion",
    "multipoles: exposed wait",
    "downward",
    "leaves",
    "output",
    "backward output",
    "total",
];

/// The number of [`STAGES`].
pub const NSTAGES: usize = 17;

/// The positions in [`STAGES`] of the compute stages, whose imbalance a [`Report`] gives:
/// upward local, downward and leaves.
pub const COMPUTE: [usize; 3] = [5, DOWNWARD, 13];

/// The position in [`STAGES`] of the downward pass.
pub const DOWNWARD: usize = 12;

/// The position in [`STAGES`] of the total.
pub const TOTAL: usize = NSTAGES - 1;

/// The operator kinds of [`PathTimes::kinds`], in `OperatorKind::ALL` order.
pub const KINDS: [&str; 8] = ["P2M", "M2M", "M2L", "P2L", "L2L", "L2P", "M2P", "P2P"];

/// The positions in [`STAGES`] of the exposed waits and of the blocking exchange stages.
const EXPOSED: [usize; 2] = [4, 11];
const EXCHANGES: [usize; 2] = [2, 9];

/// The parts of the build as a [`run`] reports them, from `BuildTimings`, then the sum
/// and the wall time of `build` on the rank; seconds.
pub const BUILD_PARTS: [&str; NBUILD] = [
    "domain",
    "octree",
    "plan",
    "second cut",
    "redistribution",
    "tables",
    "evaluator",
    "load",
    "sum",
    "wall",
];

/// The values of [`STAGES`] for one evaluation of the path `overlapped`.
fn stage_values(t: &StageTimings, overlapped: bool) -> [f64; NSTAGES] {
    let s = Duration::as_secs_f64;
    let o = t.overlap.unwrap_or_default();
    let (sources, multipoles) = if overlapped {
        (
            [s(&o.sources.total), s(&o.sources.exposed)],
            [s(&o.multipoles.total), s(&o.multipoles.exposed)],
        )
    } else {
        ([s(&t.exchange_sources); 2], [s(&t.exchange_multipoles); 2])
    };
    [
        s(&t.forward_charges),
        s(&t.load),
        s(&t.exchange_sources),
        sources[0],
        sources[1],
        s(&t.upward_local),
        s(&t.upward_global),
        s(&t.reduction),
        s(&t.replicated_top),
        s(&t.exchange_multipoles),
        multipoles[0],
        multipoles[1],
        s(&t.downward),
        s(&t.evaluate_leaves),
        s(&t.output),
        s(&t.backward_output),
        s(&t.total()),
    ]
}

/// The time of each kind of [`KINDS`] in one evaluation, seconds; NaN without kind
/// timings.
fn kind_values(t: &StageTimings) -> [f64; 8] {
    match &t.kinds {
        Some(kinds) => OperatorKind::ALL.map(|kind| kinds.get(kind).total().as_secs_f64()),
        None => [f64::NAN; 8],
    }
}

/// The details of the overlapped path as a [`run`] reports them, from `OverlapTimes`
/// (Phase 5S T9, "The exposed collectives"), seconds: the `test` calls of each exchange,
/// the multipole exchange's final wait for its sends, then its wait for each level
/// (`OverlapTimes::level_waits`, levels 0 to [`TOP_TREE_LEVELS`] − 1).
pub const OVERLAP_PARTS: [&str; 4] = [
    "sources: test calls",
    "reduction: test calls",
    "multipoles: test calls",
    "multipoles: final wait for the sends",
];

/// The number of values of [`PathTimes::overlap`] per rank: [`OVERLAP_PARTS`], then one
/// wait per level.
pub const NOVERLAP: usize = OVERLAP_PARTS.len() + TOP_TREE_LEVELS;

/// The values of [`NOVERLAP`] for one overlapped evaluation.
fn overlap_values(t: &StageTimings) -> [f64; NOVERLAP] {
    let o = t.overlap.unwrap_or_default();
    let s = Duration::as_secs_f64;
    core::array::from_fn(|k| match k {
        0 => s(&o.sources.progress),
        1 => s(&o.reduction.progress),
        2 => s(&o.multipoles.progress),
        3 => s(&o.multipoles.sends),
        _ => o.level_waits.get(k - OVERLAP_PARTS.len()).map_or(0.0, s),
    })
}

/// The number of [`BUILD_PARTS`].
pub const NBUILD: usize = 10;

/// The values of [`BUILD_PARTS`] for one build.
fn build_values(t: &BuildTimings, wall: Duration) -> [f64; NBUILD] {
    let s = Duration::as_secs_f64;
    [
        s(&t.domain),
        s(&t.octree),
        s(&t.plan),
        s(&t.cut),
        s(&t.redistribute),
        s(&t.tables),
        s(&t.evaluator),
        s(&t.load),
        s(&t.total()),
        s(&wall),
    ]
}

/// The median of `values` (the mean of the two middle ones for an even count); NaN if
/// empty or if the middle values are NaN.
pub fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// What a rank's exchanges move per evaluation, in messages and values (a value is the
/// precision's size in bytes): `Fmm::exchange_traffic` and `Fmm::exchange_sizes`, and the
/// moves of the charges and the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankTraffic {
    /// The source exchange.
    pub sources: Traffic,
    /// The upward reduction, summed over the levels (Phase 5S T6; per level in
    /// [`TopTreeRow`], T5's place for the sum).
    pub reduction: Traffic,
    /// The multipole exchange, summed over the levels.
    pub multipoles: Traffic,
    /// The values of this rank's boxes of the replicated top, which its all-gather sends
    /// to each other rank (Phase 5S T5; the coarse blocks of the coarse gather before).
    pub top_sent: usize,
    /// The values of the other ranks' boxes of the replicated top, which the all-gather
    /// delivers to this rank.
    pub top_received: usize,
    /// This rank's boxes of the replicated top and the other ranks' that the all-gather
    /// writes here (B_o, B_r).
    pub top_boxes: [usize; 2],
    /// The forward of the charges (an all-to-all-v): values sent (one per input source)
    /// and received (one per owned source).
    pub charges: [usize; 2],
    /// The backward of the output (an all-to-all-v): values sent (four per owned target)
    /// and received (four per input target).
    pub output: [usize; 2],
}

/// The memory of one rank by the formula of docs/design/distributed-fmm.md §9.2, as
/// docs/design/scale-out.md §8.2 updates it for the locally essential, owned top tree
/// (Phase 5S T6), in bytes: a *model*, not a measurement. The held boxes K are the
/// plan's, H since Phase 5S T5; the buffers hold every exchange's values, the upward
/// reduction's and the replicated top's all-gathered |R| n_c values included.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemoryModel {
    /// Multipoles and locals: 2 n_c s K, a slot for every held box (n_c = m (p + 1)² with m
    /// charge vectors, Phase 6 T5).
    pub coefficients: f64,
    /// Source store, target input and output: ((3 + m) (N_s + G_s) + 3 N_t + 4 m N_t) s.
    pub points: f64,
    /// Box index and leaf numbering: 32 K + 13 (J_loc + J_gh).
    pub index: f64,
    /// The views: 12 E_V + 4 (E_near + E_W + E_X).
    pub views: f64,
    /// The two redistributions: 4 per sent and 12 per received point each.
    pub redistribution: f64,
    /// The exchange buffers: every value sent and received, the reduction's and the
    /// replicated top's included.
    pub buffers: f64,
    /// The replicated top's keys, slots and offsets: 24 |R| (Phase 5S T5; the coarse
    /// gather's keys and blocks, 16 B, before).
    pub top: f64,
    /// The partition map (Phase 5S T3; scale-out.md §8.2): 8 B for the keys of the B
    /// blocks, B / 8 for their leaf bits and 8 (P + 1) for the cut; 8 P (the partition
    /// bounds) before Phase 5S T6.
    pub partition: f64,
    /// Under `M2lStrategy::PlaneWave` (Phase 6 T13): the waves of one frame of the largest
    /// level, 4H s per held box, and each thread's scratch,
    /// `LaplaceOperator::plane_wave_bytes`; zero otherwise.
    pub plane_waves: f64,
}

/// The number of parts of a [`MemoryModel`].
const MEMORY_PARTS: usize = 9;

impl MemoryModel {
    /// The sum of the parts.
    pub fn total(&self) -> f64 {
        self.coefficients
            + self.points
            + self.index
            + self.views
            + self.redistribution
            + self.buffers
            + self.top
            + self.partition
            + self.plane_waves
    }

    fn to_array(self) -> [f64; MEMORY_PARTS] {
        [
            self.coefficients,
            self.points,
            self.index,
            self.views,
            self.redistribution,
            self.buffers,
            self.top,
            self.partition,
            self.plane_waves,
        ]
    }

    fn from_array(a: &[f64]) -> Self {
        Self {
            coefficients: a[0],
            points: a[1],
            index: a[2],
            views: a[3],
            redistribution: a[4],
            buffers: a[5],
            top: a[6],
            partition: a[7],
            plane_waves: a[8],
        }
    }
}

/// One rank's work as built, its build times, its traffic and its memory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RankRow {
    /// The points this rank passed (sources, equal to the targets).
    pub input: usize,
    /// The sources this rank owns after the redistribution.
    pub sources: usize,
    /// The targets this rank owns.
    pub targets: usize,
    /// The local leaves.
    pub leaves: usize,
    /// The local leaves that hold no source.
    pub empty_leaves: usize,
    /// The interaction lists on this rank.
    pub lists: ListSizes,
    /// The held boxes of the plan ([`HELD_PARTS`]): own, halo, shared ancestors, the
    /// other top-tree keys of the locally essential set H, and the keys held outside H
    /// (Phase 5S T4's top-tree keys of the coarse gather and the global pass, T ∖ H;
    /// docs/design/scale-out.md §3.7; none since T5). The first four sum to |H|.
    pub held: [usize; 5],
    /// The ghost source points the source exchange receives (G_s).
    pub ghost_sources: usize,
    /// [`BUILD_PARTS`], the median over the builds.
    pub build: [f64; NBUILD],
    /// The traffic per evaluation.
    pub traffic: RankTraffic,
    /// The §9.2 formula, evaluated.
    pub memory: MemoryModel,
    /// The resident size at the end of the run, in bytes, where it can be read
    /// ([`resident_memory`]).
    pub resident: Option<f64>,
    /// The resident size after the input was generated, before the timed builds (Phase 5S
    /// T2): the harness's own memory, which must not be O(N) on any rank (but rank 0
    /// with [`Input::Rank0`]); with [`Input::Owners`] it includes the build that finds the
    /// owners.
    pub resident_input: Option<f64>,
    /// The bytes of the plan's arrays, counted from their lengths (Phase 5S T4): the box
    /// index (keys, kinds, first children, leaves), the leaf numbering and every view,
    /// row offsets and batch arrays included; what `Plan::new` keeps.
    pub plan_bytes: usize,
    /// The owned top tree (Phase 5S T5; docs/design/scale-out.md §4), per evaluation.
    pub top_tree: TopTreeRow,
    /// The work by operator kind on this rank's plan (`Fmm::kind_counts`, Phase 5S T7),
    /// in `OperatorKind::ALL` order, in the units of `nd_fmm_exec::cost`.
    pub kind_counts: [u64; 8],
    /// The modelled cost of an evaluation on this rank, Σ_k α_k c_k with the run's
    /// constants ([`Report::cost_model`]), in seconds.
    pub modelled: f64,
    /// The near rule's pairs on this rank (`Fmm::near_rule`, Phase 6 T12): the W pairs
    /// between leaves, those converted to near pairs, the X pairs between leaves, those
    /// converted; all zero with the rule off.
    pub near_pairs: [usize; 4],
    /// The W entries of this rank's local leaves after the near rule (Phase 6 T12), by
    /// [`W_ENTRY_BINS`]: interior boxes, leaves by their source points, and leaves whose
    /// count this rank does not hold (another rank's; none on one rank).
    pub w_entries: [usize; W_ENTRY_BINS.len()],
}

/// The bins of [`RankRow::w_entries`] (Phase 6 T12, docs/design/optimisation.md §8a.2): the
/// W entries that are interior boxes, the leaves with 0, 1–8, 9–16, 17–32, 33–64 and more
/// than 64 source points, and the leaves of other ranks.
pub const W_ENTRY_BINS: [&str; 8] = [
    "interior",
    "0",
    "1-8",
    "9-16",
    "17-32",
    "33-64",
    ">64",
    "other rank",
];

/// The levels a [`TopTreeRow`] counts per level: 0 to 16, the octree's deepest level.
pub const TOP_TREE_LEVELS: usize = 17;

/// A rank's part of the owned top tree (Phase 5S T5; docs/design/scale-out.md §4), and
/// what Phase 5's coarse gather and global pass did on the same tree: counted from the
/// plan, the partition map and `Fmm::exchange_traffic`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TopTreeRow {
    /// The M2M pairs of the global pass on this rank: eight per `Global` box it owns.
    pub global_m2m: usize,
    /// Phase 5's: eight per `Global` box of Phase 5, every ancestor of a coarse block of
    /// any rank, on every rank.
    pub phase5_global_m2m: usize,
    /// The values Phase 5's coarse gather received on this rank: every other rank's
    /// coarse blocks.
    pub phase5_coarse_received: usize,
    /// Per level, the messages the reduction of the level receives on this rank.
    pub reduction_messages: [usize; TOP_TREE_LEVELS],
    /// Per level, the values the reduction of the level receives on this rank.
    pub reduction_values: [usize; TOP_TREE_LEVELS],
    /// The ranks the multipole exchange receives from, over every level, each once.
    pub multipole_neighbours: usize,
    /// Of those, the ranks on other nodes ([`Placement::node`]).
    pub multipole_off_node_neighbours: usize,
    /// The other nodes the multipole exchange receives from.
    pub multipole_neighbour_nodes: usize,
    /// The messages the multipole exchange sends to ranks on other nodes, over every
    /// level.
    pub multipole_off_node_messages: usize,
    /// The most ranks the multipole exchange sends one box's multipole to, on any level
    /// (`ExchangeTraffic::multipole_fan_out`), at any stage of the forwarding tree.
    pub multipole_fan_out: usize,
    /// The stages of the multipole exchange this rank takes part in, the most over the
    /// levels (`ExchangeTraffic::multipole_depth`; Phase 5S T8s).
    pub multipole_depth: usize,
    /// The chunks this rank forwards along the forwarding tree, over every level
    /// (`ExchangeTraffic::multipole_forwarded`; Phase 5S T8s).
    pub multipole_forwarded: usize,
    /// The readers a top-tree box needs to be in the replicated top
    /// (`ExchangeTraffic::top_threshold`): ⌈P/16⌉ (at least 2), or a higher power of two
    /// where more than 4,096 boxes reach it.
    pub top_threshold: usize,
}

impl TopTreeRow {
    /// The number of values of [`encode`](Self::encode).
    const VALUES: usize = 11 + 2 * TOP_TREE_LEVELS;

    fn encode(&self) -> Vec<f64> {
        let mut values = vec![
            self.global_m2m,
            self.phase5_global_m2m,
            self.phase5_coarse_received,
            self.multipole_neighbours,
            self.multipole_off_node_neighbours,
            self.multipole_neighbour_nodes,
            self.multipole_off_node_messages,
            self.multipole_fan_out,
            self.top_threshold,
        ];
        values.extend(self.reduction_messages);
        values.extend(self.reduction_values);
        values.extend([self.multipole_depth, self.multipole_forwarded]);
        values.into_iter().map(|v| v as f64).collect()
    }

    fn decode(v: &[f64]) -> Self {
        let c = |k: usize| v[k] as usize;
        Self {
            global_m2m: c(0),
            phase5_global_m2m: c(1),
            phase5_coarse_received: c(2),
            multipole_neighbours: c(3),
            multipole_off_node_neighbours: c(4),
            multipole_neighbour_nodes: c(5),
            multipole_off_node_messages: c(6),
            multipole_fan_out: c(7),
            top_threshold: c(8),
            reduction_messages: core::array::from_fn(|l| c(9 + l)),
            reduction_values: core::array::from_fn(|l| c(9 + TOP_TREE_LEVELS + l)),
            multipole_depth: c(9 + 2 * TOP_TREE_LEVELS),
            multipole_forwarded: c(10 + 2 * TOP_TREE_LEVELS),
        }
    }
}

/// The position in [`RankRow::encode`] of the second part of the held boxes, after the
/// counts, the build, the traffic, the memory and the resident size; the third part, the
/// resident size after the input and the last two parts follow.
const ROW_VALUES: usize = 11 + NBUILD + 20 + MEMORY_PARTS + 1;

/// The position in [`RankRow::encode`] of the traffic, after the counts and the build.
const ROW_TRAFFIC: usize = 11 + NBUILD;

impl RankRow {
    /// The row as values, for an all-gather (every count is exact in f64).
    fn encode(&self) -> Vec<f64> {
        let t = &self.traffic;
        let counts = [
            self.input,
            self.sources,
            self.targets,
            self.leaves,
            self.empty_leaves,
            self.lists.u,
            self.lists.v,
            self.lists.w,
            self.lists.x,
            self.ghost_sources,
            self.held[0],
        ];
        let traffic = [
            t.sources.messages_sent,
            t.sources.messages_received,
            t.sources.values_sent,
            t.sources.values_received,
            t.multipoles.messages_sent,
            t.multipoles.messages_received,
            t.multipoles.values_sent,
            t.multipoles.values_received,
            t.reduction.messages_sent,
            t.reduction.messages_received,
            t.reduction.values_sent,
            t.reduction.values_received,
            t.top_sent,
            t.top_received,
            t.top_boxes[0],
            t.top_boxes[1],
            t.charges[0],
            t.charges[1],
            t.output[0],
            t.output[1],
        ];
        let mut values: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        values.extend(self.build);
        values.extend(traffic.iter().map(|&c| c as f64));
        values.extend(self.memory.to_array());
        values.push(self.resident.unwrap_or(-1.0));
        values.push(self.held[1] as f64);
        values.push(self.held[2] as f64);
        values.push(self.resident_input.unwrap_or(-1.0));
        values.push(self.held[3] as f64);
        values.push(self.held[4] as f64);
        values.push(self.plan_bytes as f64);
        values.extend(self.top_tree.encode());
        values.extend(self.kind_counts.map(|c| c as f64));
        values.push(self.modelled);
        values.extend(self.near_pairs.map(|c| c as f64));
        values.extend(self.w_entries.map(|c| c as f64));
        values
    }

    /// The row of [`encode`](Self::encode).
    fn decode(v: &[f64]) -> Self {
        let c = |k: usize| v[k] as usize;
        let t = |k: usize| v[ROW_TRAFFIC + k] as usize;
        let memory = ROW_TRAFFIC + 20;
        let kinds = ROW_VALUES + 6 + TopTreeRow::VALUES;
        Self {
            input: c(0),
            sources: c(1),
            targets: c(2),
            leaves: c(3),
            empty_leaves: c(4),
            lists: ListSizes {
                u: c(5),
                v: c(6),
                w: c(7),
                x: c(8),
            },
            ghost_sources: c(9),
            held: [
                c(10),
                c(ROW_VALUES),
                c(ROW_VALUES + 1),
                c(ROW_VALUES + 3),
                c(ROW_VALUES + 4),
            ],
            build: core::array::from_fn(|k| v[11 + k]),
            traffic: RankTraffic {
                sources: Traffic {
                    messages_sent: t(0),
                    messages_received: t(1),
                    values_sent: t(2),
                    values_received: t(3),
                },
                multipoles: Traffic {
                    messages_sent: t(4),
                    messages_received: t(5),
                    values_sent: t(6),
                    values_received: t(7),
                },
                reduction: Traffic {
                    messages_sent: t(8),
                    messages_received: t(9),
                    values_sent: t(10),
                    values_received: t(11),
                },
                top_sent: t(12),
                top_received: t(13),
                top_boxes: [t(14), t(15)],
                charges: [t(16), t(17)],
                output: [t(18), t(19)],
            },
            memory: MemoryModel::from_array(&v[memory..memory + MEMORY_PARTS]),
            resident: (v[memory + MEMORY_PARTS] >= 0.0).then_some(v[memory + MEMORY_PARTS]),
            resident_input: (v[ROW_VALUES + 2] >= 0.0).then_some(v[ROW_VALUES + 2]),
            plan_bytes: c(ROW_VALUES + 5),
            top_tree: TopTreeRow::decode(&v[ROW_VALUES + 6..kinds]),
            kind_counts: core::array::from_fn(|k| v[kinds + k] as u64),
            modelled: v[kinds + 8],
            near_pairs: core::array::from_fn(|k| v[kinds + 9 + k] as usize),
            w_entries: core::array::from_fn(|k| v[kinds + 13 + k] as usize),
        }
    }
}

/// The stage times of one evaluation path.
#[derive(Clone, Debug, PartialEq)]
pub struct PathTimes {
    /// Whether the exchanges overlapped.
    pub overlapped: bool,
    /// Per rank, the median of each of [`STAGES`] over the evaluations.
    pub medians: Vec<[f64; NSTAGES]>,
    /// The median over the evaluations of the maximum over the ranks of the total: the
    /// evaluation's wall time.
    pub wall: f64,
    /// With [`Settings::kinds`], per rank the median of each of [`KINDS`] over the
    /// evaluations, seconds; empty otherwise.
    pub kinds: Vec<[f64; 8]>,
    /// In the overlapped path, per rank the median over the evaluations of each value of
    /// [`NOVERLAP`] (Phase 5S T9), seconds; empty in the blocking path.
    pub overlap: Vec<[f64; NOVERLAP]>,
}

impl PathTimes {
    /// The maximum, minimum and mean over the ranks of stage `k`.
    pub fn reduced(&self, k: usize) -> [f64; 3] {
        reduce_over(self.medians.iter().map(|m| m[k]))
    }

    /// The maximum, minimum and mean over the ranks of kind `k` of [`KINDS`]; NaN without
    /// kind timings.
    pub fn kind(&self, k: usize) -> [f64; 3] {
        reduce_over(self.kinds.iter().map(|t| t[k]))
    }

    /// The maximum, minimum and mean over the ranks of value `k` of [`NOVERLAP`]; NaN in
    /// the blocking path.
    pub fn overlap_part(&self, k: usize) -> [f64; 3] {
        reduce_over(self.overlap.iter().map(|t| t[k]))
    }

    /// The time per V pair, in seconds, of the rank whose `time` is the largest: its
    /// `time` over its V pairs (`rows` in rank order); NaN if `time` is NaN everywhere.
    pub fn per_v_pair(&self, rows: &[RankRow], time: impl Fn(usize) -> f64) -> f64 {
        (0..rows.len())
            .filter(|&r| !time(r).is_nan())
            .max_by(|&a, &b| time(a).total_cmp(&time(b)))
            .map_or(f64::NAN, |r| time(r) / rows[r].lists.v.max(1) as f64)
    }

    /// The maximum, minimum and mean over the ranks of the sum of the [`COMPUTE`] stages.
    pub fn compute(&self) -> [f64; 3] {
        reduce_over(
            self.medians
                .iter()
                .map(|m| COMPUTE.iter().map(|&k| m[k]).sum()),
        )
    }

    /// The maximum over the ranks of the exposed waits of both exchanges together, and
    /// of the two exchange stages together: in an overlapped path against a blocking
    /// one, the C5.2 measure (docs/design/distributed-fmm.md §8.6).
    pub fn exchanges(&self) -> (f64, f64) {
        let max = |ks: [usize; 2]| {
            self.medians
                .iter()
                .map(|m| m[ks[0]] + m[ks[1]])
                .fold(0.0, f64::max)
        };
        (max(EXPOSED), max(EXCHANGES))
    }
}

/// The cost model of `nd_fmm_exec::cost` against the measured kind times of a path
/// (Phase 5S T7; docs/design/scale-out.md §6.1): the constants fitted to them, and how
/// well the model reproduces every rank's compute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelFit {
    /// The fitted constants α_k, in ns per counted unit, in [`KINDS`] order: the time of
    /// kind k summed over the ranks over its count summed over the ranks (0 for a kind
    /// without a count).
    pub alpha: [f64; 8],
    /// The largest relative difference over the ranks between the modelled compute with
    /// the fitted constants and the measured one (the sum of the kind times).
    pub error_max: f64,
    /// The measured compute's max over mean (the sum of the kind times per rank).
    pub measured_imbalance: f64,
    /// The modelled compute's max over mean, with the fitted constants.
    pub fitted_imbalance: f64,
    /// The modelled compute's max over mean, with the run's constants
    /// ([`RankRow::modelled`]).
    pub modelled_imbalance: f64,
}

impl PathTimes {
    /// The fit of the cost model to this path's kind times and the ranks' counts; `None`
    /// without kind timings.
    pub fn fit(&self, rows: &[RankRow]) -> Option<ModelFit> {
        if self.kinds.is_empty() {
            return None;
        }
        let alpha: [f64; 8] = core::array::from_fn(|k| {
            let time: f64 = self.kinds.iter().map(|t| t[k]).sum();
            let count: f64 = rows.iter().map(|r| r.kind_counts[k] as f64).sum();
            if count > 0.0 { 1e9 * time / count } else { 0.0 }
        });
        let measured: Vec<f64> = self.kinds.iter().map(|t| t.iter().sum()).collect();
        let fitted: Vec<f64> = rows
            .iter()
            .map(|r| {
                (0..8)
                    .map(|k| 1e-9 * alpha[k] * r.kind_counts[k] as f64)
                    .sum()
            })
            .collect();
        let imbalance = |values: &[f64]| {
            let [max, _, mean] = reduce_over(values.iter().copied());
            max / mean
        };
        let error_max = measured
            .iter()
            .zip(&fitted)
            .filter(|(m, _)| **m > 0.0)
            .map(|(m, f)| (f - m).abs() / m)
            .fold(0.0, f64::max);
        let modelled: Vec<f64> = rows.iter().map(|r| r.modelled).collect();
        Some(ModelFit {
            alpha,
            error_max,
            measured_imbalance: imbalance(&measured),
            fitted_imbalance: imbalance(&fitted),
            modelled_imbalance: imbalance(&modelled),
        })
    }
}

/// The maximum, minimum and mean of `values`, NaN ignored (all NaN if every value is).
fn reduce_over(values: impl Iterator<Item = f64>) -> [f64; 3] {
    let values: Vec<f64> = values.filter(|v| !v.is_nan()).collect();
    if values.is_empty() {
        return [f64::NAN; 3];
    }
    [
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().sum::<f64>() / values.len() as f64,
    ]
}

/// The errors against the direct sum (module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Errors {
    /// The charge vectors.
    pub vectors: usize,
    /// The sampled targets.
    pub sampled: usize,
    /// Root mean squares over the vectors of the relative L2 and max errors of φ.
    pub potential: ErrorNorms,
    /// The same of ∇φ.
    pub gradient: ErrorNorms,
    /// The most pairs a rank's part of the direct sum evaluates per vector: its sources
    /// times the sampled targets, O(N/P) (Phase 5S T2).
    pub direct_pairs: usize,
    /// The time of a rank's part of the direct sum, summed over the vectors, on the
    /// slowest rank, in seconds; not part of the evaluation times.
    pub direct_seconds: f64,
}

/// The output of charge vector 0 against the one-rank `Fmm` of the same settings over
/// every point in index order, built and evaluated on rank 0 alone, over every target:
/// within 100 u_T on any rank count (docs/design/distributed-fmm.md §5), zero on one rank
/// with the points in index order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reference {
    /// Relative L2 difference of φ.
    pub potential_l2: f64,
    /// Relative L2 difference of ∇φ (the Euclidean norm per target).
    pub gradient_l2: f64,
    /// The largest difference of a value over the largest magnitude of a value, over φ
    /// and the gradient components.
    pub max_relative: f64,
    /// Values that differ in any bit.
    pub differing: usize,
}

/// The result of one [`run`], the same on every rank apart from
/// [`machine`](Self::machine), which rank 0 fills.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// The machine: CPU, cores, host name, target and toolchain; on rank 0 only.
    pub machine: String,
    /// The problem.
    pub workload: Workload,
    /// The settings.
    pub settings: Settings,
    /// "f32" or "f64".
    pub precision: &'static str,
    /// The resolved M2L strategy.
    pub strategy: M2lStrategy,
    /// The resolved P2P kernel (`Fmm::p2p_kernel`): `Auto` replaced by its ISA.
    pub p2p: P2pChoice,
    /// The number of ranks.
    pub ranks: usize,
    /// Rank 0's threading report: rayon threads, the MPI level and the BLAS variables.
    pub threading: ThreadingReport,
    /// The number of levels.
    pub nlevels: usize,
    /// Every rank's row, in rank order.
    pub rows: Vec<RankRow>,
    /// Each path's stage times, blocking first.
    pub paths: Vec<PathTimes>,
    /// With [`Overlap::Both`], the values of the two paths that differ in any bit, over
    /// every rank and evaluation (0: overlap is bit for bit the blocking path).
    pub differing: Option<usize>,
    /// The errors, if [`Settings::vectors`] > 0.
    pub errors: Option<Errors>,
    /// The reference check, if [`Settings::reference`].
    pub reference: Option<Reference>,
    /// How [`RankRow::resident`] was read.
    pub resident_kind: &'static str,
    /// Where the ranks ran: nodes, ranks per node, hosts and CPUs, the MPI library and
    /// the Slurm job (Phase 5N T2).
    pub placement: Placement,
    /// The constants of the modelled cost (`Fmm::cost_model`, Phase 5S T7): those the
    /// second cut weighed with, and those of [`RankRow::modelled`].
    pub cost_model: CostModel,
    /// Whether the second cut moved blocks (`Fmm::recut`): it ran and gained at least
    /// `WORK_CUT_MIN_GAIN`.
    pub recut: bool,
    /// The near rule's crossovers [N_W, N_X] (`Fmm::near_rule`, Phase 6 T12); `None` with
    /// the rule off.
    pub near_crossovers: Option<[f64; 2]>,
    /// With a device backend, the bytes of its translations' scratch on this rank
    /// (`DeviceReport::scratch_bytes`, Phase 6 T7); `None` on the host.
    pub device_scratch_bytes: Option<u64>,
}

/// The resident size of this process in bytes and how it was read: the peak (`VmHWM`) from
/// `/proc/self/status` on Linux; elsewhere the current resident size from `ps -o rss=`
/// (macOS reports no peak without a new dependency). `None` if neither can be read (for
/// example `ps` inside the macOS sandbox).
pub fn resident_memory() -> Option<(f64, &'static str)> {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let kib = status
            .lines()
            .find(|l| l.starts_with("VmHWM:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())?;
        return Some((kib * 1024.0, "peak resident (VmHWM, /proc/self/status)"));
    }
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib: f64 = String::from_utf8(out.stdout).ok()?.trim().parse().ok()?;
    Some((
        kib * 1024.0,
        "resident at the end of the run (ps -o rss), not the peak",
    ))
}

/// The machine line of a report: CPU model, cores, host name, target and toolchain.
pub fn machine() -> String {
    let host = std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string());
    format!(
        "{} ({}), host {host}, {}, {}",
        cpu_model(),
        cores(),
        target(),
        toolchain()
    )
}

/// The Slurm job a run is part of, from the environment of rank 0 (Phase 5N T2): what a
/// report on a cluster names beside its figures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlurmJob {
    /// `SLURM_JOB_ID`.
    pub id: String,
    /// `SLURM_JOB_QOS`, if set.
    pub qos: Option<String>,
    /// `SLURM_JOB_PARTITION`, if set.
    pub partition: Option<String>,
    /// `SLURM_JOB_NODELIST`, if set.
    pub nodelist: Option<String>,
    /// The binding as Slurm reports it: `SLURM_CPU_BIND` (set inside an `srun` step with
    /// `--cpu-bind`), with `SLURM_CPU_BIND_LIST` if set; `None` when Slurm did not bind
    /// (for example `mpirun` inside the allocation, which binds itself).
    pub cpu_bind: Option<String>,
}

impl SlurmJob {
    /// The job of the variables that `var` returns, if `SLURM_JOB_ID` is among them.
    /// [`from_env`](Self::from_env) reads the process's environment; tests pass their own.
    pub fn from_variables(var: impl Fn(&str) -> Option<String>) -> Option<Self> {
        let id = var("SLURM_JOB_ID").filter(|id| !id.is_empty())?;
        let cpu_bind = var("SLURM_CPU_BIND").map(|bind| match var("SLURM_CPU_BIND_LIST") {
            Some(list) => format!("{bind} (list {list})"),
            None => bind,
        });
        Some(Self {
            id,
            qos: var("SLURM_JOB_QOS"),
            partition: var("SLURM_JOB_PARTITION"),
            nodelist: var("SLURM_JOB_NODELIST"),
            cpu_bind,
        })
    }

    /// The job of this process's environment, if it runs under Slurm.
    pub fn from_env() -> Option<Self> {
        Self::from_variables(|name| std::env::var(name).ok())
    }
}

impl fmt::Display for SlurmJob {
    /// For example `job 238836, QoS test, partition kathleen, nodes node-a[01-02];
    /// binding by Slurm: not set (the launcher binds)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let or = |v: &Option<String>| v.clone().unwrap_or_else(|| "unknown".to_string());
        write!(
            f,
            "job {}, QoS {}, partition {}, nodes {}; binding by Slurm: {}",
            self.id,
            or(&self.qos),
            or(&self.partition),
            or(&self.nodelist),
            self.cpu_bind
                .clone()
                .unwrap_or_else(|| "not set (the launcher binds)".to_string())
        )
    }
}

/// Where the ranks of a run ran (Phase 5N T2): the nodes, from one `split_shared` of the
/// communicator, each rank's host and CPUs, the MPI library and the Slurm job.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    /// The shared-memory domains (nodes) of the communicator.
    pub nodes: usize,
    /// The ranks of each node, the nodes in the order of their lowest rank.
    pub ranks_per_node: Vec<usize>,
    /// Per rank, its node in that order.
    pub node: Vec<usize>,
    /// Per rank, the processor name MPI gives (the host).
    pub hosts: Vec<String>,
    /// Per rank, the CPUs it may run on as the kernel reports them ([`cpus_allowed`]),
    /// "–" where they cannot be read.
    pub cpus: Vec<String>,
    /// The MPI library (`MPI_Get_library_version`, its first line) on rank 0.
    pub mpi_library: String,
    /// The Slurm job, if rank 0 runs in one.
    pub slurm: Option<SlurmJob>,
}

impl Placement {
    /// The fewest and the most ranks on a node.
    pub fn ranks_per_node_range(&self) -> (usize, usize) {
        let min = self.ranks_per_node.iter().copied().min().unwrap_or(0);
        let max = self.ranks_per_node.iter().copied().max().unwrap_or(0);
        (min, max)
    }
}

/// The CPUs this process may run on, `Cpus_allowed_list` of `/proc/self/status` (Linux),
/// for example "0-19" or "3"; `None` elsewhere. Rayon's threads inherit it, so it is the
/// binding of the whole rank.
pub fn cpus_allowed() -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .map(|v| v.trim().to_string())
}

/// The [`Placement`] of the ranks of `comm`.
///
/// # Collective operation
///
/// On every rank of `comm`: one `split_shared`, a broadcast on each node, two
/// all-gathers and one all-gather-v.
/// They run after the timed evaluations (requirement 5 of docs/phase5n/README.md:
/// counted here, and in the harness only).
fn placement(comm: &SimpleCommunicator) -> Placement {
    let size = comm.size() as usize;
    // With the rank as key, a node's rank 0 is its lowest rank, which names the node.
    let node = comm.split_shared(comm.rank());
    let mut leader = comm.rank() as u64;
    node.process_at_rank(0).broadcast_into(&mut leader);
    drop(node);
    let mut leader_of = vec![0u64; size];
    comm.all_gather_into(&leader, &mut leader_of[..]);
    let mut leaders = leader_of.clone();
    leaders.sort_unstable();
    leaders.dedup();
    let node_of: Vec<usize> = leader_of
        .iter()
        .map(|l| leaders.binary_search(l).expect("every leader is listed"))
        .collect();
    let mut ranks_per_node = vec![0usize; leaders.len()];
    for &k in &node_of {
        ranks_per_node[k] += 1;
    }
    let host = mpi::environment::processor_name().unwrap_or_else(|_| "unknown".to_string());
    let cpus = cpus_allowed().unwrap_or_else(|| "–".to_string());
    let text = format!("{host}\n{cpus}");
    let gathered = gather_strings(text.as_bytes(), comm);
    let (hosts, cpus) = gathered
        .iter()
        .map(|s| {
            let (h, c) = s.split_once('\n').unwrap_or((s.as_str(), "–"));
            (h.to_string(), c.to_string())
        })
        .unzip();
    let mpi_library = mpi::environment::library_version()
        .ok()
        .and_then(|v| {
            v.trim_matches(char::from(0))
                .lines()
                .next()
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    Placement {
        nodes: leaders.len(),
        ranks_per_node,
        node: node_of,
        hosts,
        cpus,
        mpi_library,
        slurm: SlurmJob::from_env(),
    }
}

/// Every rank's `bytes`, in rank order, as strings (an all-gather of the lengths and an
/// all-gather-v).
fn gather_strings(bytes: &[u8], comm: &SimpleCommunicator) -> Vec<String> {
    let size = comm.size() as usize;
    let length = bytes.len() as i32;
    let mut lengths = vec![0i32; size];
    comm.all_gather_into(&length, &mut lengths[..]);
    let offsets: Vec<i32> = lengths
        .iter()
        .scan(0, |acc, &c| {
            let o = *acc;
            *acc += c;
            Some(o)
        })
        .collect();
    let mut all = vec![0u8; lengths.iter().sum::<i32>() as usize];
    comm.all_gather_varcount_into(
        bytes,
        &mut PartitionMut::new(&mut all[..], &lengths[..], &offsets[..]),
    );
    (0..size)
        .map(|r| {
            let (o, l) = (offsets[r] as usize, lengths[r] as usize);
            String::from_utf8_lossy(&all[o..o + l]).into_owned()
        })
        .collect()
}

/// The positions of the points that rank `rank` of `size` passes under `input`, ascending,
/// found from the indices alone, so that any rank can produce any rank's input (Phase 5S
/// T2): every input but [`Input::Owners`], which needs a build. O(N) time for the share
/// (every index hashed), O(N/P) for the ranges; O(N/P) memory but on rank 0 with
/// [`Input::Rank0`].
///
/// # Panics
///
/// With [`Input::Owners`].
pub fn input_by_index(workload: &Workload, input: Input, rank: usize, size: usize) -> Vec<usize> {
    let n = workload.n;
    match input {
        Input::Rank0 => (0..n).filter(|_| rank == 0).collect(),
        Input::Share => (0..n)
            .filter(|&i| workload.share(i, size) == rank)
            .collect(),
        Input::Ranges => workload.range(rank, size).collect(),
        Input::Owners => panic!("the owners' input needs a build, not only the indices"),
    }
}

/// [`input_by_index`] of ranks `first` … `first + count − 1` of `size`, in rank order; the
/// share in one pass over the indices.
fn inputs_by_index(
    workload: &Workload,
    input: Input,
    (first, count): (usize, usize),
    size: usize,
) -> Vec<Vec<usize>> {
    match input {
        Input::Share => {
            let mut inputs = vec![Vec::new(); count];
            for i in 0..workload.n {
                let r = workload.share(i, size);
                if (first..first + count).contains(&r) {
                    inputs[r - first].push(i);
                }
            }
            inputs
        }
        _ => (first..first + count)
            .map(|r| input_by_index(workload, input, r, size))
            .collect(),
    }
}

/// The positions of the points this rank passes under `input`, ascending.
///
/// # Collective operation
///
/// With [`Input::Owners`], a build of the FMM from the share on every rank of `comm`.
fn input_points<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    input: Input,
    builder: &FmmBuilder<T>,
    comm: &SimpleCommunicator,
) -> Vec<usize> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let n = workload.n;
    match input {
        Input::Owners => {
            let share = input_by_index(workload, Input::Share, rank, size);
            let points: Vec<[f64; 3]> = share.iter().map(|&i| workload.point(i)).collect();
            let fmm: Fmm<'_, T> = builder
                .clone()
                .build(&points, &points, comm)
                .unwrap_or_else(|error| panic!("rank {rank}: the FMM does not build: {error}"));
            let owned = owned_points(&fmm, workload);
            let mut total = 0usize;
            comm.all_reduce_into(&owned.len(), &mut total, SystemOperation::sum());
            assert_eq!(
                total, n,
                "rank {rank}: the owners' input misses or repeats points"
            );
            owned
        }
        other => input_by_index(workload, other, rank, size),
    }
}

/// The positions of the workload's points that lie in a local leaf of `fmm`, ascending:
/// each point's index on every level of a local leaf, from the domain's box, looked up
/// among the local leaves and confirmed by `geometry::contains`, the octree's own test.
fn owned_points<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    workload: &Workload,
) -> Vec<usize> {
    let domain = *fmm.domain();
    let leaves = fmm.plan().index().leaves();
    let mut by_index = HashMap::new();
    let mut levels = BTreeSet::new();
    for j in 0..leaves.nlocal() {
        let (key, level) = (leaves.key(j), leaves.level(j));
        let index = integer_centre(key, level).map(|c| (c - 1) / 2);
        by_index.insert((level, index), key);
        levels.insert(level);
    }
    let bounds = domain.physical_box().coordinates();
    (0..workload.n)
        .filter(|&i| {
            let x = workload.point(i);
            levels.iter().any(|&level| {
                let scale = (1u64 << level) as f64;
                let index: [i64; 3] = core::array::from_fn(|k| {
                    ((x[k] - bounds[k]) / (bounds[k + 3] - bounds[k]) * scale).floor() as i64
                });
                by_index
                    .get(&(level, index))
                    .is_some_and(|&key| contains(key, x, &domain))
            })
        })
        .collect()
}

/// The parts of [`RankRow::held`].
pub const HELD_PARTS: [&str; 5] = [
    "own",
    "halo",
    "shared ancestors",
    "other top keys in H",
    "outside H (Phase 5S T4's top keys; none since T5)",
];

/// The bytes of a partition map of `blocks` blocks cut over `ranks` ranks (Phase 5S T3;
/// docs/design/scale-out.md §8.2): 8 per block for its key, a leaf bit per block (in u64
/// words) and 8 (P + 1) for the cut.
fn partition_map_bytes(blocks: usize, ranks: usize) -> f64 {
    (8 * blocks + 8 * blocks.div_ceil(64) + 8 * (ranks + 1)) as f64
}

/// The held boxes of `fmm`'s plan on this rank, by [`HELD_PARTS`]: a key of the octree's
/// locally essential set H is own (non-ghost there, not `Global`), a shared ancestor
/// (`Global` there), a top-tree key of another rank (a ghost that the partition map
/// knows: another rank's block, or an ancestor of other ranks' blocks only) or halo (a
/// ghost below another rank's block); every other held key is outside H (Phase 5S T4's
/// top-tree keys of the coarse gather and the global pass; none since T5).
fn held_boxes<T: Stored + SimdScalar + Equivalence + Default>(fmm: &Fmm<'_, T>) -> [usize; 5] {
    let index = fmm.plan().index();
    let octree = fmm.octree();
    let essential = octree.all_keys();
    let map = octree.partition_map();
    let mut held = [0usize; 5];
    for level in 0..index.nlevels() {
        for &key in index.keys(level) {
            let k = match essential.get(&key) {
                None => 4,
                // `KeyType` is nd-octree's, which this crate does not depend on: its
                // `Global` variant is told apart by name.
                Some(kind) if !kind.is_ghost() && format!("{kind:?}") == "Global" => 2,
                Some(kind) if !kind.is_ghost() => 0,
                Some(_) if map.top_kind(key).is_some() => 3,
                Some(_) => 1,
            };
            held[k] += 1;
        }
    }
    held
}

/// This rank's row before the build times and the resident size.
fn rank_row<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    input: usize,
    p: usize,
    ranks: usize,
) -> RankRow {
    let (sources, targets) = fmm.owned_points();
    let m = fmm.vectors();
    let traffic = fmm.exchange_traffic();
    let sizes = fmm.exchange_sizes();
    let multipoles = traffic
        .multipoles
        .iter()
        .fold(Traffic::default(), |sum, t| Traffic {
            messages_sent: sum.messages_sent + t.messages_sent,
            messages_received: sum.messages_received + t.messages_received,
            values_sent: sum.values_sent + t.values_sent,
            values_received: sum.values_received + t.values_received,
        });
    let reduction = traffic
        .reduction
        .iter()
        .fold(Traffic::default(), |sum, t| Traffic {
            messages_sent: sum.messages_sent + t.messages_sent,
            messages_received: sum.messages_received + t.messages_received,
            values_sent: sum.values_sent + t.values_sent,
            values_received: sum.values_received + t.values_received,
        });
    let rank_traffic = RankTraffic {
        sources: traffic.sources,
        reduction,
        multipoles,
        top_sent: traffic.top_sent,
        top_received: traffic.top_received,
        top_boxes: [sizes.sent_top, sizes.received_top],
        charges: [m * input, m * sources],
        output: [m * OUTPUT_VALUES * targets, m * OUTPUT_VALUES * input],
    };
    let held = held_boxes(fmm);
    let counts = fmm.kind_counts();
    let k: usize = held.iter().sum();
    let leaves = fmm.plan().index().leaves();
    let (j_loc, j_all) = (leaves.nlocal(), leaves.len());
    let (mut e_v, mut e_rest) = (0usize, 0usize);
    for lists in fmm.plan().levels() {
        e_v += lists.v().len();
        e_rest += lists.near().len() + lists.w().len() + lists.x().len();
    }
    let s = size_of::<T>() as f64;
    // The values of a box's multipole: m (p + 1)² with m charge vectors (Phase 6 T5).
    let n_c = (m * (p + 1) * (p + 1)) as f64;
    let ghost_sources = sizes.ghost_sources;
    let top_tree = top_tree_row(fmm, &traffic, n_c as usize);
    let values = rank_traffic.sources.values_sent
        + rank_traffic.sources.values_received
        + multipoles.values_sent
        + multipoles.values_received
        + reduction.values_sent
        + reduction.values_received
        + traffic.top_sent
        + traffic.top_received;
    let memory = MemoryModel {
        coefficients: 2.0 * n_c * s * k as f64,
        points: ((3 + m) * (sources + ghost_sources) + (3 + m * OUTPUT_VALUES) * targets) as f64
            * s,
        index: (32 * k + 13 * j_all) as f64,
        views: (12 * e_v + 4 * e_rest) as f64,
        redistribution: 2.0 * (4 * input + 12 * sources) as f64,
        buffers: values as f64 * s,
        top: 24.0 * ((traffic.top_sent + traffic.top_received) as f64 / n_c),
        partition: partition_map_bytes(fmm.octree().partition_map().blocks().len(), ranks),
        plane_waves: fmm.operator().plane_wave_bytes() as f64,
    };
    RankRow {
        input,
        sources,
        targets,
        leaves: j_loc,
        empty_leaves: fmm.source_counts().iter().filter(|&&c| c == 0).count(),
        lists: fmm.list_sizes(),
        held,
        ghost_sources,
        build: [0.0; NBUILD],
        traffic: rank_traffic,
        memory,
        resident: None,
        resident_input: None,
        plan_bytes: plan_bytes(fmm),
        top_tree,
        kind_counts: counts.to_array(),
        modelled: fmm.cost_model().cost(&counts) * 1e-9,
        w_entries: w_entries(fmm),
        near_pairs: {
            let pairs = fmm.near_rule().pairs;
            [
                pairs.w_leaf_pairs,
                pairs.w_converted,
                pairs.x_leaf_pairs,
                pairs.x_converted,
            ]
        },
    }
}

/// This rank's [`TopTreeRow`]: the owned top tree from `fmm`'s plan and `traffic`, and
/// Phase 5's coarse gather and global pass on the same tree from the partition map, with
/// `n_c` values per multipole.
fn top_tree_row<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    traffic: &ExchangeTraffic,
    n_c: usize,
) -> TopTreeRow {
    let map = fmm.octree().partition_map();
    // Phase 5's `Global` boxes, every ancestor of a block of any rank: the blocks are the
    // leaves of a complete tree whose interior boxes have eight children each, so B blocks
    // have (B − 1) / 7 ancestors.
    let blocks = map.blocks().len();
    let own = map.rank_blocks(map.rank()).len();
    let mut row = TopTreeRow {
        global_m2m: fmm
            .plan()
            .levels()
            .iter()
            .map(|l| l.m2m_global().len())
            .sum(),
        phase5_global_m2m: 8 * (blocks - 1) / 7,
        phase5_coarse_received: (blocks - own) * n_c,
        multipole_neighbours: traffic.multipole_neighbours.len(),
        multipole_fan_out: traffic.multipole_fan_out,
        multipole_depth: traffic.multipole_depth,
        multipole_forwarded: traffic.multipole_forwarded,
        top_threshold: traffic.top_threshold,
        ..TopTreeRow::default()
    };
    for (level, t) in traffic.reduction.iter().enumerate().take(TOP_TREE_LEVELS) {
        row.reduction_messages[level] = t.messages_received;
        row.reduction_values[level] = t.values_received;
    }
    row
}

/// The name of `routing` in the summary and the report: `direct`, or `tree` with its
/// fan-out (`tree8`).
pub fn routing_name(routing: MultipoleRouting) -> String {
    match routing {
        MultipoleRouting::Direct => "direct".to_owned(),
        MultipoleRouting::Tree { fan_out } => format!("tree{fan_out}"),
    }
}

/// This rank's W entries by [`W_ENTRY_BINS`], from its plan and the source counts of its
/// local leaves.
fn w_entries<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
) -> [usize; W_ENTRY_BINS.len()] {
    let plan = fmm.plan();
    let index = plan.index();
    let counts = fmm.source_counts();
    let mut bins = [0usize; W_ENTRY_BINS.len()];
    for level in 0..plan.nlevels().saturating_sub(1) {
        for &s in plan.level(level).w().entries() {
            // `KeyType` is nd-octree's, which this crate does not depend on: a ghost leaf
            // by its name.
            let kind = format!("{:?}", index.kind(level + 1, s as usize));
            let bin = match index.box_leaf(level + 1, s as usize) {
                None if kind.starts_with("GhostLeaf") => 7,
                None => 0,
                Some(j) => match counts.get(j as usize) {
                    None => 7,
                    Some(0) => 1,
                    Some(1..=8) => 2,
                    Some(9..=16) => 3,
                    Some(17..=32) => 4,
                    Some(33..=64) => 5,
                    Some(_) => 6,
                },
            };
            bins[bin] += 1;
        }
    }
    bins
}

/// The near rule by name, as `--near` takes it (Phase 6 T12): `off`, `cost`, or the
/// crossovers `W,X`.
pub fn near_name(rule: NearRuleChoice) -> String {
    match rule {
        NearRuleChoice::Off => "off".to_owned(),
        NearRuleChoice::CostModel => "cost".to_owned(),
        NearRuleChoice::Crossovers { w, x } => format!("{w},{x}"),
    }
}

/// Fills `row`'s counts of the multipole exchange by node: the neighbours on other nodes,
/// those nodes, and the messages sent to them, from `traffic` and every rank's node
/// (`node`, [`Placement::node`]).
fn off_node_counts(row: &mut TopTreeRow, traffic: &ExchangeTraffic, node: &[usize], rank: usize) {
    let off = |r: &&usize| node[**r] != node[rank];
    row.multipole_off_node_neighbours = traffic.multipole_neighbours.iter().filter(off).count();
    let mut nodes: Vec<usize> = traffic
        .multipole_neighbours
        .iter()
        .filter(off)
        .map(|&r| node[r])
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    row.multipole_neighbour_nodes = nodes.len();
    row.multipole_off_node_messages = traffic
        .multipole_destinations
        .iter()
        .map(|ranks| ranks.iter().filter(off).count())
        .sum();
}

/// The bytes of the arrays of `fmm`'s plan, from their lengths ([`RankRow::plan_bytes`]).
fn plan_bytes<T: Stored + SimdScalar + Equivalence + Default>(fmm: &Fmm<'_, T>) -> usize {
    use std::mem::size_of_val;
    let plan = fmm.plan();
    let index = plan.index();
    let leaves = index.leaves();
    let mut bytes =
        size_of_val(leaves.keys()) + size_of_val(leaves.levels()) + size_of_val(leaves.boxes());
    for level in 0..index.nlevels() {
        bytes += size_of_val(index.keys(level))
            + size_of_val(index.kinds(level))
            + size_of_val(index.first_children(level))
            + size_of_val(index.box_leaves(level));
    }
    // The views are named by their accessors: this crate does not depend on nd-fmm-plan.
    macro_rules! csr {
        ($view:expr) => {
            size_of_val($view.row_offsets()) + size_of_val($view.entries())
        };
    }
    macro_rules! grouped {
        ($view:expr) => {
            size_of_val($view.row_offsets())
                + size_of_val($view.sources())
                + size_of_val($view.groups())
                + size_of_val($view.batch_offsets())
                + size_of_val($view.batch_targets())
                + size_of_val($view.batch_sources())
        };
    }
    for lists in plan.levels() {
        bytes += grouped!(lists.v())
            + grouped!(lists.m2m_local())
            + grouped!(lists.m2m_global())
            + grouped!(lists.l2l())
            + csr!(lists.x())
            + csr!(lists.p2m())
            + csr!(lists.near())
            + csr!(lists.w())
            + csr!(lists.l2p());
    }
    bytes + size_of_val(plan.coarse_blocks())
}

/// The bits of every value of `output` of several vectors (Phase 6 T5), φ then ∇φ.
fn bits_many<T: RealScalar>(output: &OutputMany<T>) -> Vec<u64> {
    let mut bits: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        bits.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    bits
}

/// The bits of every value of `output`, potentials then gradients.
fn bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut bits: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        bits.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    bits
}

/// The builder of `settings`: the harness's tree, gradients, the threads, the strategy, its
/// node budget and the batched M2L's mode, the P2P kernel, overlap, the kind timings, the second cut and
/// its constants.
fn builder<T: Stored + SimdScalar + Equivalence + Default>(settings: &Settings) -> FmmBuilder<T> {
    let mut builder = FmmBuilder::<T>::new(settings.p)
        .max_level(MAX_LEVEL)
        .max_points_per_leaf(MAX_POINTS_PER_LEAF)
        .gradients(true)
        .threads(settings.threads)
        .strategy(settings.strategy)
        .m2l_mul_add(settings.m2l_mul_add)
        .p2p_kernel(settings.p2p)
        .overlap(settings.overlap == Overlap::On)
        .kind_timings(if settings.kinds {
            KindTiming::Synchronous
        } else {
            KindTiming::Off
        });
    if let Some(bytes) = settings.dense_node_budget {
        builder = builder.dense_node_budget(bytes);
    }
    builder = builder
        .work_cut(settings.work_cut)
        .multipole_routing(settings.routing)
        .vectors(settings.rhs)
        .backend(settings.backend)
        .device_gemm(settings.device_gemm)
        .device_gather(settings.device_gather)
        .device_reduction(settings.device_reduction);
    if let Some(bytes) = settings.device_scratch {
        builder = builder.device_scratch_budget(bytes);
    }
    if let Some(model) = settings.cost_model {
        builder = builder.cost_model(model);
    }
    builder.near_rule(settings.near_rule)
}

/// Builds the FMM of `workload` in precision `T` with `settings` on every rank of
/// `comm`, evaluates it and measures it (module documentation).
///
/// # Collective operation
///
/// On every rank of `comm`, with the same arguments; every rank returns the same
/// [`Report`] apart from [`Report::machine`].
///
/// # Errors
///
/// [`HarnessError::ReferenceTooLarge`] with [`Settings::reference`] above
/// [`REFERENCE_MAX_POINTS`] points, on every rank before any collective.
///
/// # Panics
///
/// If the FMM does not build or evaluate (for example with `threads` > 1 when MPI
/// provides less than `Threading::Funneled`), or with [`Input::Owners`] if the points
/// found on their owners are not every point once.
pub fn run<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    settings: &Settings,
    comm: &SimpleCommunicator,
) -> Result<Report, HarnessError> {
    if settings.reference && workload.n > REFERENCE_MAX_POINTS {
        return Err(HarnessError::ReferenceTooLarge {
            n: workload.n,
            max: REFERENCE_MAX_POINTS,
        });
    }
    if settings.rhs > 1 && settings.vectors > 0 {
        return Err(HarnessError::SeveralVectors("--errors"));
    }
    if settings.rhs > 1 && settings.reference {
        return Err(HarnessError::SeveralVectors("--reference"));
    }
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let builder = builder::<T>(settings);
    let mine = input_points(workload, settings.input, &builder, comm);
    let points: Vec<[f64; 3]> = mine.iter().map(|&i| workload.point(i)).collect();
    // The harness's own memory: the input, before any build that is timed.
    let resident_input = resident_memory().map(|(bytes, _)| bytes);

    // The builds: the times of each, the last kept.
    let mut fmm: Option<Fmm<'_, T>> = None;
    let mut builds = Vec::new();
    for _ in 0..settings.builds.max(1) {
        drop(fmm.take());
        comm.barrier();
        let start = Instant::now();
        let built = builder
            .clone()
            .build(&points, &points, comm)
            .unwrap_or_else(|error| panic!("rank {rank}: the FMM does not build: {error}"));
        builds.push(build_values(&built.build_timings(), start.elapsed()));
        fmm = Some(built);
    }
    let mut fmm = fmm.expect("at least one build");
    let mut row = rank_row(&fmm, mine.len(), settings.p, size);
    row.build = core::array::from_fn(|k| median(builds.iter().map(|b| b[k]).collect()));
    row.resident_input = resident_input;

    // The timed evaluations of charge vector 0 (with `rhs` vectors, of vectors 0 to
    // rhs − 1 at once), the paths alternating.
    let all_charges: Vec<Vec<T>> = (0..settings.rhs.max(1))
        .map(|k| {
            mine.iter()
                .map(|&i| T::from_f64(workload.charge(k, i)))
                .collect()
        })
        .collect();
    let charges = &all_charges[0];
    let several: Vec<&[T]> = all_charges.iter().map(Vec::as_slice).collect();
    let paths = settings.overlap.paths();
    let mut times: Vec<Vec<[f64; NSTAGES]>> = vec![Vec::new(); paths.len()];
    let mut kinds: Vec<Vec<[f64; 8]>> = vec![Vec::new(); paths.len()];
    let mut overlap: Vec<Vec<[f64; NOVERLAP]>> = vec![Vec::new(); paths.len()];
    let mut differing = 0usize;
    let mut last = None;
    for e in 0..=settings.evaluations {
        let mut outputs = Vec::new();
        for (path, &overlapped) in paths.iter().enumerate() {
            fmm.set_overlap(overlapped);
            comm.barrier();
            let (timings, output, output_bits) = if settings.rhs > 1 {
                let output = fmm.evaluate_many(&several).unwrap_or_else(|error| {
                    panic!("rank {rank}: the FMM does not evaluate: {error}")
                });
                (output.timings, None, bits_many(&output))
            } else {
                let output = fmm.evaluate(charges).unwrap_or_else(|error| {
                    panic!("rank {rank}: the FMM does not evaluate: {error}")
                });
                let output_bits = bits(&output);
                (output.timings, Some(output), output_bits)
            };
            if e > 0 {
                times[path].push(stage_values(&timings, overlapped));
                kinds[path].push(kind_values(&timings));
                overlap[path].push(overlap_values(&timings));
            }
            outputs.push((output, output_bits));
        }
        if let [blocking, overlapped] = &outputs[..] {
            differing += blocking
                .1
                .iter()
                .zip(&overlapped.1)
                .filter(|(a, b)| a != b)
                .count();
        }
        last = outputs.pop().and_then(|(output, _)| output);
    }

    // The stage medians and the totals of every evaluation, gathered.
    let mut path_times = Vec::new();
    for (path, &overlapped) in paths.iter().enumerate() {
        let local: [f64; NSTAGES] =
            core::array::from_fn(|k| median(times[path].iter().map(|t| t[k]).collect()));
        let mut every = vec![0.0f64; NSTAGES * size];
        comm.all_gather_into(&local[..], &mut every[..]);
        let totals: Vec<f64> = times[path].iter().map(|t| t[TOTAL]).collect();
        let mut all_totals = vec![0.0f64; totals.len() * size];
        comm.all_gather_into(&totals[..], &mut all_totals[..]);
        let evaluations = totals.len();
        let wall = median(
            (0..evaluations)
                .map(|e| {
                    (0..size)
                        .map(|r| all_totals[r * evaluations + e])
                        .fold(0.0, f64::max)
                })
                .collect(),
        );
        let kind_medians = if settings.kinds {
            let local: [f64; 8] =
                core::array::from_fn(|k| median(kinds[path].iter().map(|t| t[k]).collect()));
            let mut every = vec![0.0f64; 8 * size];
            comm.all_gather_into(&local[..], &mut every[..]);
            every.as_chunks::<8>().0.to_vec()
        } else {
            Vec::new()
        };
        let overlap_medians = if overlapped {
            let local: [f64; NOVERLAP] =
                core::array::from_fn(|k| median(overlap[path].iter().map(|t| t[k]).collect()));
            let mut every = vec![0.0f64; NOVERLAP * size];
            comm.all_gather_into(&local[..], &mut every[..]);
            every.as_chunks::<NOVERLAP>().0.to_vec()
        } else {
            Vec::new()
        };
        path_times.push(PathTimes {
            overlapped,
            medians: every.as_chunks::<NSTAGES>().0.to_vec(),
            wall,
            kinds: kind_medians,
            overlap: overlap_medians,
        });
    }
    let differing = (paths.len() == 2).then(|| {
        let mut total = 0usize;
        comm.all_reduce_into(&differing, &mut total, SystemOperation::sum());
        total
    });

    let errors = (settings.vectors > 0)
        .then(|| errors(workload, settings, (&mine, &points), &mut fmm, comm));
    let reference = settings.reference.then(|| {
        let strategy = settings.reference_strategy.unwrap_or(fmm.strategy());
        let builder = builder.clone().strategy(strategy);
        let last = last
            .as_ref()
            .expect("one vector, the warm-up's output at least");
        reference(workload, &builder, (&mine, &points), last, comm)
    });

    // Every rank's row, with the resident size at the end and the multipole exchange's
    // neighbours by node.
    let placement = placement(comm);
    off_node_counts(
        &mut row.top_tree,
        &fmm.exchange_traffic(),
        &placement.node,
        rank,
    );
    let resident = resident_memory();
    row.resident = resident.map(|(bytes, _)| bytes);
    let local = row.encode();
    let mut every = vec![0.0f64; local.len() * size];
    comm.all_gather_into(&local[..], &mut every[..]);
    let rows = every
        .chunks_exact(local.len())
        .map(RankRow::decode)
        .collect();

    Ok(Report {
        machine: if rank == 0 { machine() } else { String::new() },
        workload: *workload,
        settings: *settings,
        precision: if size_of::<T>() == 4 { "f32" } else { "f64" },
        strategy: fmm.strategy(),
        p2p: fmm.p2p_kernel(),
        ranks: size,
        threading: fmm.threading().clone(),
        nlevels: fmm.nlevels(),
        rows,
        paths: path_times,
        differing,
        errors,
        reference,
        resident_kind: resident.map_or("not available", |(_, kind)| kind),
        placement,
        cost_model: fmm.cost_model(),
        recut: fmm.recut(),
        near_crossovers: fmm.near_rule().crossovers,
        device_scratch_bytes: device_scratch_bytes(&fmm),
    })
}

/// The bytes of the device translations' scratch of `fmm` (Phase 6 T7), `None` on the host.
#[cfg(feature = "gpu")]
fn device_scratch_bytes<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
) -> Option<u64> {
    fmm.device_report().map(|report| report.scratch_bytes)
}

/// Without the device path there is no device scratch.
#[cfg(not(feature = "gpu"))]
fn device_scratch_bytes<T: Stored + SimdScalar + Equivalence + Default>(
    _fmm: &Fmm<'_, T>,
) -> Option<u64> {
    None
}

/// The errors of `fmm` against the direct sum (module documentation): `settings.vectors`
/// evaluations and the direct sum over the ranks.
fn errors<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    settings: &Settings,
    (mine, points): (&[usize], &[[f64; 3]]),
    fmm: &mut Fmm<'_, T>,
    comm: &SimpleCommunicator,
) -> Errors {
    let sample = workload.sample(settings.sampled.min(workload.n));
    let targets: Vec<[f64; 3]> = sample.iter().map(|&i| workload.point(i)).collect();
    // The sampled targets this rank passed: their places in the sample and in `mine`.
    let local: Vec<(usize, usize)> = sample
        .iter()
        .enumerate()
        .filter_map(|(s, i)| mine.binary_search(i).ok().map(|r| (s, r)))
        .collect();
    let scale = 4.0 * std::f64::consts::PI;
    let (mut phi_l2, mut phi_max, mut grad_l2, mut grad_max) = (0.0, 0.0, 0.0, 0.0);
    let mut direct_seconds = 0.0;
    for vector in 0..settings.vectors {
        let charges: Vec<T> = mine
            .iter()
            .map(|&i| T::from_f64(workload.charge(vector, i)))
            .collect();
        let rounded: Vec<f64> = charges.iter().map(|&q| RealScalar::to_f64(q)).collect();
        // The direct sum over this rank's sources, then over every rank.
        let mut phi = vec![0.0; targets.len()];
        let mut grad = vec![[0.0; 3]; targets.len()];
        let start = Instant::now();
        direct_sum(points, &rounded, &targets, &mut phi, Some(&mut grad));
        direct_seconds += start.elapsed().as_secs_f64();
        let (mut exact_phi, mut exact_grad) = (vec![0.0; phi.len()], vec![0.0; 3 * phi.len()]);
        comm.all_reduce_into(&phi[..], &mut exact_phi[..], SystemOperation::sum());
        comm.all_reduce_into(
            grad.as_flattened(),
            &mut exact_grad[..],
            SystemOperation::sum(),
        );
        let output = fmm
            .evaluate(&charges)
            .unwrap_or_else(|error| panic!("the FMM does not evaluate: {error}"));
        let gradient = output.gradient.as_ref().expect("built with gradients");
        let (mut potential_errors, mut gradient_errors) =
            (ErrorAccumulator::new(), ErrorAccumulator::new());
        for &(s, r) in &local {
            potential_errors.add_values(&output.potential[r..=r], &[exact_phi[s] / scale]);
            let g: [f64; 3] = core::array::from_fn(|c| exact_grad[3 * s + c] / scale);
            gradient_errors.add_vectors(&gradient[r..=r], &[g]);
        }
        let p = reduce(&potential_errors, comm).finish();
        let g = reduce(&gradient_errors, comm).finish();
        phi_l2 += p.l2 * p.l2;
        phi_max += p.max * p.max;
        grad_l2 += g.l2 * g.l2;
        grad_max += g.max * g.max;
    }
    let rms = |sum: f64| (sum / settings.vectors as f64).sqrt();
    let local = [(points.len() * targets.len()) as f64, direct_seconds];
    let mut most = [0.0f64; 2];
    comm.all_reduce_into(&local[..], &mut most[..], SystemOperation::max());
    Errors {
        vectors: settings.vectors,
        sampled: sample.len(),
        direct_pairs: most[0] as usize,
        direct_seconds: most[1],
        potential: ErrorNorms {
            l2: rms(phi_l2),
            max: rms(phi_max),
        },
        gradient: ErrorNorms {
            l2: rms(grad_l2),
            max: rms(grad_max),
        },
    }
}

/// The accumulator of every rank's `local`: its sums added and its maxima taken over the
/// ranks (two all-reduces).
fn reduce(local: &ErrorAccumulator, comm: &SimpleCommunicator) -> ErrorAccumulator {
    let parts = local.parts();
    let (mut sums, mut maxima) = ([0.0f64; 2], [0.0f64; 2]);
    comm.all_reduce_into(&parts[..2], &mut sums[..], SystemOperation::sum());
    comm.all_reduce_into(&parts[2..], &mut maxima[..], SystemOperation::max());
    ErrorAccumulator::from_parts([sums[0], sums[1], maxima[0], maxima[1]])
}

/// The [`Reference`] check of `output` (charge vector 0): every rank's output gathered on
/// rank 0 with its indices, against the one-rank `Fmm` of `builder` over every point in
/// index order, which rank 0 builds on its own communicator; the result broadcast.
fn reference<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    builder: &FmmBuilder<T>,
    (mine, points): (&[usize], &[[f64; 3]]),
    output: &Output<T>,
    comm: &SimpleCommunicator,
) -> Reference {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let root = comm.process_at_rank(0);
    let gradient = output.gradient.as_ref().expect("built with gradients");
    let indices: Vec<u64> = mine.iter().map(|&i| i as u64).collect();
    let values: Vec<f64> = (0..mine.len())
        .flat_map(|r| {
            let g = gradient[r].map(RealScalar::to_f64);
            [RealScalar::to_f64(output.potential[r]), g[0], g[1], g[2]]
        })
        .collect();
    let count = mine.len() as i32;
    let mut result = [0.0f64; 4];
    if rank == 0 {
        let mut counts = vec![0i32; size];
        root.gather_into_root(&count, &mut counts[..]);
        let offsets: Vec<i32> = counts
            .iter()
            .scan(0, |acc, &c| {
                let o = *acc;
                *acc += c;
                Some(o)
            })
            .collect();
        let total = counts.iter().sum::<i32>() as usize;
        let mut all_indices = vec![0u64; total];
        root.gather_varcount_into_root(
            &indices[..],
            &mut PartitionMut::new(&mut all_indices[..], &counts[..], &offsets[..]),
        );
        let counts4: Vec<i32> = counts.iter().map(|&c| 4 * c).collect();
        let offsets4: Vec<i32> = offsets.iter().map(|&o| 4 * o).collect();
        let mut all_values = vec![0.0f64; 4 * total];
        root.gather_varcount_into_root(
            &values[..],
            &mut PartitionMut::new(&mut all_values[..], &counts4[..], &offsets4[..]),
        );
        // The one-rank FMM over every point in index order.
        let every: Vec<[f64; 3]> = if mine.len() == workload.n {
            points.to_vec()
        } else {
            (0..workload.n).map(|i| workload.point(i)).collect()
        };
        let charges: Vec<T> = (0..workload.n)
            .map(|i| T::from_f64(workload.charge(0, i)))
            .collect();
        let one = SimpleCommunicator::self_comm();
        let mut fmm: Fmm<'_, T> = builder
            .clone()
            .build(&every, &every, &one)
            .unwrap_or_else(|error| panic!("the one-rank FMM does not build: {error}"));
        let exact = fmm
            .evaluate(&charges)
            .unwrap_or_else(|error| panic!("the one-rank FMM does not evaluate: {error}"));
        let exact_gradient = exact.gradient.as_ref().expect("built with gradients");
        let (mut phi, mut grad) = (ErrorAccumulator::new(), ErrorAccumulator::new());
        let (mut value_max, mut difference_max, mut differing) = (0.0f64, 0.0f64, 0usize);
        for (k, &i) in all_indices.iter().enumerate() {
            let i = i as usize;
            let got = &all_values[4 * k..4 * k + 4];
            let want = [
                RealScalar::to_f64(exact.potential[i]),
                RealScalar::to_f64(exact_gradient[i][0]),
                RealScalar::to_f64(exact_gradient[i][1]),
                RealScalar::to_f64(exact_gradient[i][2]),
            ];
            phi.add_values(&got[..1], &want[..1]);
            grad.add_vectors(&[[got[1], got[2], got[3]]], &[[want[1], want[2], want[3]]]);
            for (a, b) in got.iter().zip(want) {
                value_max = value_max.max(b.abs());
                difference_max = difference_max.max((a - b).abs());
                differing += usize::from(a.to_bits() != b.to_bits());
            }
        }
        assert_eq!(total, workload.n, "every target once");
        result = [
            phi.finish().l2,
            grad.finish().l2,
            difference_max / value_max,
            differing as f64,
        ];
    } else {
        root.gather_into(&count);
        root.gather_varcount_into(&indices[..]);
        root.gather_varcount_into(&values[..]);
    }
    root.broadcast_into(&mut result[..]);
    Reference {
        potential_l2: result[0],
        gradient_l2: result[1],
        max_relative: result[2],
        differing: result[3] as usize,
    }
}

/// One output's values at a rank's targets, φ and ∇φ flattened, for [`invariance`].
struct Values<T> {
    potential: Vec<T>,
    gradient: Vec<T>,
}

impl<T: Copy> Values<T> {
    fn new(output: Output<T>) -> Self {
        let gradient = output.gradient.expect("built with gradients");
        Self {
            potential: output.potential,
            gradient: gradient.as_flattened().to_vec(),
        }
    }

    /// The values of `count` targets from `first`, φ then ∇φ: one message.
    fn slice(&self, first: usize, count: usize) -> Vec<T> {
        let mut values = self.potential[first..first + count].to_vec();
        values.extend_from_slice(&self.gradient[3 * first..3 * (first + count)]);
        values
    }

    /// Every value, φ then ∇φ.
    fn all(&self) -> Vec<T> {
        self.slice(0, self.potential.len())
    }
}

/// Builds the FMM of `builder` on `comm` over the points `indices` of `workload` and
/// evaluates charge vector 0 once per path of `overlapped`. Returns the outputs in that
/// order and the resolved strategy; the FMM is dropped before it returns.
///
/// # Collective operation
///
/// On every rank of `comm`.
fn evaluate_paths<T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    workload: &Workload,
    indices: &[usize],
    overlapped: &[bool],
    comm: &SimpleCommunicator,
) -> (Vec<Values<T>>, M2lStrategy) {
    let rank = comm.rank();
    let points: Vec<[f64; 3]> = indices.iter().map(|&i| workload.point(i)).collect();
    let charges: Vec<T> = indices
        .iter()
        .map(|&i| T::from_f64(workload.charge(0, i)))
        .collect();
    let mut fmm: Fmm<'_, T> = builder
        .build(&points, &points, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the FMM does not build: {error}"));
    let outputs =
        overlapped
            .iter()
            .map(|&overlap| {
                fmm.set_overlap(overlap);
                Values::new(fmm.evaluate(&charges).unwrap_or_else(|error| {
                    panic!("rank {rank}: the FMM does not evaluate: {error}")
                }))
            })
            .collect();
    (outputs, fmm.strategy())
}

/// The values of `a` and `b` that differ in any bit (compared in f64, which holds an f32
/// exactly).
fn differing_values<T: RealScalar>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len(), "the same targets");
    a.iter()
        .zip(b)
        .filter(|&(&x, &y)| RealScalar::to_f64(x).to_bits() != RealScalar::to_f64(y).to_bits())
        .count()
}

/// Receives `values` from rank `from` of `comm`, testing the request every millisecond and
/// sleeping in between, so that a rank that waits for a small run of [`invariance`] leaves
/// its core to the small run's ranks when the ranks oversubscribe the cores (a blocking
/// receive spins).
fn receive_sleeping<T: Equivalence>(comm: &SimpleCommunicator, from: usize, values: &mut [T]) {
    mpi::request::scope(|scope| {
        let mut request = comm
            .process_at_rank(from as i32)
            .immediate_receive_into(scope, values);
        loop {
            match request.test() {
                Ok(_) => break,
                Err(pending) => {
                    request = pending;
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    });
}

/// One check of [`invariance`]: the run on P ranks against the run on P/k ranks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvarianceCheck {
    /// The factor k.
    pub k: usize,
    /// The ranks of the small run, P / k.
    pub small_ranks: usize,
    /// The values compared, over every rank: four per target (φ and ∇φ) and path of the
    /// large run.
    pub compared: usize,
    /// The values that differ in any bit, summed over the ranks.
    pub differing: usize,
    /// The most values that differ on one rank.
    pub differing_max: usize,
    /// The ranks with a value that differs.
    pub ranks_differing: usize,
    /// The lowest rank with a value that differs, if one has.
    pub first_rank_differing: Option<usize>,
    /// The most points a rank of the small run passed (k inputs concatenated).
    pub small_input_max: usize,
    /// The build and evaluation of the small run on its slowest rank, in seconds.
    pub small_seconds: f64,
    /// The check from the split to the comparison on the slowest rank, in seconds.
    pub seconds: f64,
}

/// The result of [`invariance`], the same on every rank apart from
/// [`machine`](Self::machine), which rank 0 fills.
#[derive(Clone, Debug, PartialEq)]
pub struct Invariance {
    /// The machine: CPU, cores, host name, target and toolchain; on rank 0 only.
    pub machine: String,
    /// The problem.
    pub workload: Workload,
    /// The settings.
    pub settings: Settings,
    /// "f32" or "f64".
    pub precision: &'static str,
    /// The strategy of every run: the one [`Settings::strategy`] names, or under `Auto` the
    /// one the large run resolved, named in the small runs.
    pub strategy: M2lStrategy,
    /// The ranks of the large run, P.
    pub ranks: usize,
    /// The most points a rank of the large run passed.
    pub input_max: usize,
    /// The paths of the large run (overlapped or not), compared each with the small run's.
    pub large_paths: Vec<bool>,
    /// The path of the small runs: overlapped unless [`Settings::overlap`] is
    /// [`Overlap::Off`].
    pub small_overlapped: bool,
    /// The build and evaluations of the large run on its slowest rank, in seconds.
    pub large_seconds: f64,
    /// One check per factor, in the order given.
    pub checks: Vec<InvarianceCheck>,
    /// The largest resident size of a rank at the end ([`resident_memory`]), in bytes.
    pub resident_max: Option<f64>,
    /// How the resident size was read.
    pub resident_kind: &'static str,
    /// Where the ranks ran.
    pub placement: Placement,
}

/// Rank-count invariance at scale (Phase 5S T2, C5S.1; requirement 3 of
/// docs/phase5s/README.md; docs/design/scale-out.md §5.1): the FMM of `workload` with
/// `settings` on the P ranks of `comm` against the same FMM on P/k ranks, for every k of
/// `factors`, bit for bit, when rank r of the small run holds the concatenation of the
/// inputs of ranks k r, …, k r + k − 1 of the large one, in that order.
///
/// Every rank finds its input from the indices ([`input_by_index`]). The large run builds
/// and evaluates charge vector 0 on every path of [`Settings::overlap`], keeps its outputs
/// and drops the FMM. For each k, `split` of `comm` gives the small communicator: colour 0
/// on the ranks k r, none elsewhere; the other ranks enter nothing of it. Small rank r
/// finds the inputs of ranks k r … k r + k − 1 (the share in one pass over the indices),
/// concatenates them, builds the FMM with the strategy of the large run named (`Auto`
/// resolves by the ranks per node, which differ), evaluates one path (overlapped unless
/// overlap is off), and sends each rank k r + i (i ≥ 1) its slice of the output, point to
/// point; the sizes are known on both sides, and a waiting rank sleeps between tests of its
/// receive, so that oversubscribed small runs get the cores. Each rank compares its own
/// targets' values with every path of its large run, and four all-reduces agree the
/// counts. Nothing is
/// gathered: a rank holds its large input and outputs, and a small rank k inputs.
///
/// # Collective operation
///
/// On every rank of `comm`, with the same arguments; every rank returns the same
/// [`Invariance`] apart from [`Invariance::machine`].
///
/// # Errors
///
/// On every rank, before any collective: [`HarnessError::Factor`] for a factor that is 0
/// or does not divide the ranks, [`HarnessError::InputNotByIndex`] with
/// [`Input::Owners`].
///
/// # Panics
///
/// If an FMM does not build or evaluate.
pub fn invariance<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    settings: &Settings,
    factors: &[usize],
    comm: &SimpleCommunicator,
) -> Result<Invariance, HarnessError> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    if settings.input == Input::Owners {
        return Err(HarnessError::InputNotByIndex(settings.input));
    }
    if let Some(&k) = factors.iter().find(|&&k| k == 0 || !size.is_multiple_of(k)) {
        return Err(HarnessError::Factor { k, ranks: size });
    }
    if settings.rhs > 1 {
        return Err(HarnessError::SeveralVectors("--invariance"));
    }
    let builder = builder::<T>(settings);

    // The large run.
    comm.barrier();
    let start = Instant::now();
    let mine = input_by_index(workload, settings.input, rank, size);
    let large_paths = settings.overlap.paths().to_vec();
    let (large, strategy) = evaluate_paths(&builder, workload, &mine, &large_paths, comm);
    let large_seconds = start.elapsed().as_secs_f64();
    let builder = builder.strategy(strategy);
    let small_overlapped = settings.overlap != Overlap::Off;
    let large_all: Vec<Vec<T>> = large.iter().map(Values::all).collect();
    drop(large);

    let mut checks = Vec::with_capacity(factors.len());
    for &k in factors {
        comm.barrier();
        let start = Instant::now();
        let leader = rank / k * k;
        let color = if rank == leader {
            Color::with_value(0)
        } else {
            Color::undefined()
        };
        let small_comm = comm.split_by_color(color);
        let (mut small_input, mut small_seconds) = (0usize, 0.0f64);
        // This rank's slice of the small run's output.
        let small: Vec<T> = match &small_comm {
            Some(small) => {
                let started = Instant::now();
                let inputs = inputs_by_index(workload, settings.input, (leader, k), size);
                let concatenated = inputs.concat();
                small_input = concatenated.len();
                let (outputs, resolved) = evaluate_paths(
                    &builder,
                    workload,
                    &concatenated,
                    &[small_overlapped],
                    small,
                );
                assert_eq!(resolved, strategy, "rank {rank}: the strategy named");
                small_seconds = started.elapsed().as_secs_f64();
                let mut first = inputs[0].len();
                for (i, input) in inputs.iter().enumerate().skip(1) {
                    comm.process_at_rank((leader + i) as i32)
                        .send(&outputs[0].slice(first, input.len())[..]);
                    first += input.len();
                }
                outputs[0].slice(0, inputs[0].len())
            }
            None => {
                let mut values = vec![T::default(); OUTPUT_VALUES * mine.len()];
                receive_sleeping(comm, leader, &mut values);
                values
            }
        };
        drop(small_comm);
        let local: usize = large_all
            .iter()
            .map(|values| differing_values(values, &small))
            .sum();
        let seconds = start.elapsed().as_secs_f64();
        let mut sums = [0usize; 2];
        comm.all_reduce_into(
            &[usize::from(local > 0), large_all.len() * small.len()][..],
            &mut sums[..],
            SystemOperation::sum(),
        );
        let mut differing = 0usize;
        comm.all_reduce_into(&local, &mut differing, SystemOperation::sum());
        let mut most = [0.0f64; 4];
        comm.all_reduce_into(
            &[local as f64, small_input as f64, small_seconds, seconds][..],
            &mut most[..],
            SystemOperation::max(),
        );
        let mut first = 0usize;
        comm.all_reduce_into(
            &(if local > 0 { rank } else { usize::MAX }),
            &mut first,
            SystemOperation::min(),
        );
        checks.push(InvarianceCheck {
            k,
            small_ranks: size / k,
            compared: sums[1],
            differing,
            differing_max: most[0] as usize,
            ranks_differing: sums[0],
            first_rank_differing: (first != usize::MAX).then_some(first),
            small_input_max: most[1] as usize,
            small_seconds: most[2],
            seconds: most[3],
        });
    }

    let mut most = [0.0f64; 3];
    let resident = resident_memory();
    comm.all_reduce_into(
        &[
            mine.len() as f64,
            large_seconds,
            resident.map_or(-1.0, |(bytes, _)| bytes),
        ][..],
        &mut most[..],
        SystemOperation::max(),
    );
    let placement = placement(comm);
    Ok(Invariance {
        machine: if rank == 0 { machine() } else { String::new() },
        workload: *workload,
        settings: *settings,
        precision: if size_of::<T>() == 4 { "f32" } else { "f64" },
        strategy,
        ranks: size,
        input_max: most[0] as usize,
        large_paths,
        small_overlapped,
        large_seconds: most[1],
        checks,
        resident_max: (most[2] >= 0.0).then_some(most[2]),
        resident_kind: resident.map_or("not available", |(_, kind)| kind),
        placement,
    })
}

impl Invariance {
    /// The values that differ over every check.
    pub fn differing(&self) -> usize {
        self.checks.iter().map(|c| c.differing).sum()
    }

    /// One `key=value` line per check, for the job tables.
    pub fn summaries(&self) -> Vec<String> {
        self.checks
            .iter()
            .map(|c| {
                let mut fields = vec![
                    format!("dist={}", self.workload.distribution.name()),
                    format!("n={}", self.workload.n),
                    format!("precision={}", self.precision),
                    format!("p={}", self.settings.p),
                    format!("strategy={:?}", self.strategy),
                    format!("requested={:?}", self.settings.strategy),
                    format!("input={}", self.settings.input.name()),
                    format!("overlap={}", self.settings.overlap.name()),
                    format!("threads={}", self.settings.threads),
                    format!("ranks={}", self.ranks),
                    format!("small_ranks={}", c.small_ranks),
                    format!("k={}", c.k),
                    format!("compared={}", c.compared),
                    format!("differing={}", c.differing),
                    format!("differing_max={}", c.differing_max),
                    format!("ranks_differing={}", c.ranks_differing),
                    format!("small_input_max={}", c.small_input_max),
                    format!("nodes={}", self.placement.nodes),
                ];
                if let Some(job) = &self.placement.slurm {
                    fields.push(format!("job={}", job.id));
                }
                format!("invariance: {}", fields.join(" "))
            })
            .collect()
    }
}

impl fmt::Display for Invariance {
    /// The report in Markdown: the heading, the settings and placement, one row per check,
    /// the verdict and the [`summaries`](Invariance::summaries).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.settings;
        let path = |overlapped: bool| if overlapped { "overlapped" } else { "blocking" };
        let smalls: Vec<String> = self
            .checks
            .iter()
            .map(|c| c.small_ranks.to_string())
            .collect();
        writeln!(
            f,
            "## invariance: {} N = {}, {} p = {}, {} ranks against {}",
            self.workload.distribution.name(),
            self.workload.n,
            self.precision,
            s.p,
            self.ranks,
            smalls.join(", ")
        )?;
        writeln!(f)?;
        writeln!(f, "- machine: {}", self.machine)?;
        writeln!(
            f,
            "- problem: N = {}, {}; seed {:#x}, every point generated by index; max_level \
             {MAX_LEVEL}, {MAX_POINTS_PER_LEAF} points per leaf; charge vector 0, gradients",
            self.workload.n,
            self.workload.distribution.description(),
            self.workload.seed
        )?;
        writeln!(
            f,
            "- precision {}, p = {}, strategy {:?} in every run ({}); input: {}; threads per \
             rank {}",
            self.precision,
            s.p,
            self.strategy,
            if s.strategy == M2lStrategy::Auto {
                "Auto as resolved by the large run, named in the small runs".to_string()
            } else {
                format!("{:?} as named", s.strategy)
            },
            s.input.description(),
            s.threads
        )?;
        let large: Vec<&str> = self.large_paths.iter().map(|&o| path(o)).collect();
        writeln!(
            f,
            "- the large run on {} ranks evaluates {}, each compared bit for bit with the small \
             run's {} output at every target; small rank r holds the inputs of ranks k r … \
             k r + k − 1 concatenated, and sends each its slice point to point (nothing \
             gathered)",
            self.ranks,
            large.join(" and "),
            path(self.small_overlapped)
        )?;
        let placement = &self.placement;
        let (fewest, most) = placement.ranks_per_node_range();
        writeln!(
            f,
            "- placement: {} node(s), ranks per node {}; MPI library: {}",
            placement.nodes,
            if fewest == most {
                format!("{most}")
            } else {
                format!("{fewest}–{most}")
            },
            placement.mpi_library
        )?;
        if let Some(job) = &placement.slurm {
            writeln!(f, "- Slurm: {job}")?;
        }
        writeln!(
            f,
            "- the large run: at most {} points per rank, {:.2} s to build and evaluate on the \
             slowest rank; largest resident size of a rank at the end {} MB ({}); times are \
             for the record (oversubscribed runs are never timings)",
            self.input_max,
            self.large_seconds,
            self.resident_max.map_or("–".to_string(), mb),
            self.resident_kind
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "| P | P / k | k | values compared | differing (sum over ranks) | most on a rank | \
             ranks with any | first rank with any | points per small rank (max) | small run, \
             s | check, s |"
        )?;
        writeln!(f, "| {} |", ["---"; 11].join(" | "))?;
        for c in &self.checks {
            writeln!(
                f,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {:.2} |",
                self.ranks,
                c.small_ranks,
                c.k,
                c.compared,
                c.differing,
                c.differing_max,
                c.ranks_differing,
                c.first_rank_differing
                    .map_or("–".to_string(), |r| r.to_string()),
                c.small_input_max,
                c.small_seconds,
                c.seconds
            )?;
        }
        writeln!(f)?;
        let differing = self.differing();
        if differing == 0 {
            writeln!(
                f,
                "Rank-count invariance: 0 values differ ({} ranks against {})",
                self.ranks,
                smalls.join(", ")
            )?;
        } else {
            writeln!(
                f,
                "Rank-count invariance FAILED: {differing} values differ (table above)"
            )?;
        }
        writeln!(f)?;
        for line in self.summaries() {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

/// Milliseconds with three decimals, or "–" for NaN.
fn ms(seconds: f64) -> String {
    if seconds.is_nan() {
        "–".to_string()
    } else {
        format!("{:.3}", seconds * 1e3)
    }
}

/// Megabytes with one decimal.
fn mb(bytes: f64) -> String {
    format!("{:.1}", bytes / 1e6)
}

/// Above this rank count the per-rank tables are left out (the reduced ones stay).
pub const PER_RANK_ROWS: usize = 128;

impl Report {
    /// The private dense tables of `ranks` ranks at this p and precision, in bytes:
    /// `ranks` × 332 (p + 1)⁴ values (`NodeContext::dense_node_bytes`), whatever the
    /// strategy.
    fn dense_bytes(&self, ranks: usize) -> f64 {
        let value_bytes = if self.precision == "f32" { 4 } else { 8 };
        NodeContext::new(self.settings.p, value_bytes, ranks, 0).dense_node_bytes() as f64
    }

    /// The heading of the report: the problem, the ranks and the settings.
    fn heading(&self) -> String {
        format!(
            "{} N = {}, {} p = {}, {} ranks x {} thread(s), input {}",
            self.workload.distribution.name(),
            self.workload.n,
            self.precision,
            self.settings.p,
            self.ranks,
            self.settings.threads,
            self.settings.input.name()
        )
    }

    /// One `key=value` line per path for the sweeps' tables: the problem, the settings,
    /// the wall time, the build, the stage maxima and the compute imbalance, the exchange
    /// times, the errors and the memory.
    pub fn summaries(&self) -> Vec<String> {
        let rows = &self.rows;
        let max = |f: &dyn Fn(&RankRow) -> f64| rows.iter().map(f).fold(0.0, f64::max);
        let build_max: Vec<f64> = (0..BUILD_PARTS.len())
            .map(|k| max(&|r: &RankRow| r.build[k]))
            .collect();
        let resident = rows
            .iter()
            .filter_map(|r| r.resident)
            .fold(f64::NAN, f64::max);
        let resident_input = rows
            .iter()
            .filter_map(|r| r.resident_input)
            .fold(f64::NAN, f64::max);
        self.paths
            .iter()
            .map(|path| {
                let compute = path.compute();
                let (exposed, exchanges) = path.exchanges();
                let mut fields = vec![
                    format!("dist={}", self.workload.distribution.name()),
                    format!("n={}", self.workload.n),
                    format!("precision={}", self.precision),
                    format!("p={}", self.settings.p),
                    format!("ranks={}", self.ranks),
                    format!("threads={}", self.settings.threads),
                    format!("input={}", self.settings.input.name()),
                    format!("overlap={}", if path.overlapped { "on" } else { "off" }),
                    format!("strategy={:?}", self.strategy),
                    format!("requested={:?}", self.settings.strategy),
                    format!("mul_add={}", self.settings.m2l_mul_add),
                    format!("p2p={}", self.p2p),
                    format!(
                        "budget_mb={}",
                        mb(self
                            .settings
                            .dense_node_budget
                            .unwrap_or(M2lStrategy::AUTO_DENSE_NODE_BYTES)
                            as f64)
                    ),
                    format!("dense_rank_mb={}", mb(self.dense_bytes(1))),
                    format!(
                        "dense_node_mb={}",
                        mb(self.dense_bytes(self.threading.ranks_per_node))
                    ),
                    format!("wall_ms={}", ms(path.wall)),
                    format!("rhs={}", self.settings.rhs),
                    format!("backend={}", self.settings.backend),
                    format!(
                        "device_scratch_mb={}",
                        self.settings
                            .device_scratch
                            .map_or("default".to_owned(), |b| (b >> 20).to_string())
                    ),
                    format!("device_gemm={:?}", self.settings.device_gemm),
                    format!("device_gather={:?}", self.settings.device_gather),
                    format!("device_reduction={:?}", self.settings.device_reduction),
                    format!(
                        "device_scratch_bytes={}",
                        self.device_scratch_bytes
                            .map_or("none".to_owned(), |b| b.to_string())
                    ),
                    format!("build_ms={}", ms(build_max[9])),
                    format!("redistribution_build_ms={}", ms(build_max[4])),
                    format!("cut_build_ms={}", ms(build_max[3])),
                ];
                for (k, key) in [
                    (0, "forward"),
                    (5, "upward_local"),
                    (6, "upward_global"),
                    (7, "reduction"),
                    (8, "replicated_top"),
                    (DOWNWARD, "downward"),
                    (13, "leaves"),
                    (15, "backward"),
                    (TOTAL, "total"),
                ] {
                    fields.push(format!("{key}_max_ms={}", ms(path.reduced(k)[0])));
                }
                fields.push(format!("tables_ms={}", ms(build_max[5])));
                // The second cut (Phase 5S T7): the balance of the work and of the model.
                let imbalance = |get: &dyn Fn(&RankRow) -> f64| {
                    let [max, _, mean] = reduce_over(rows.iter().map(get));
                    max / mean
                };
                fields.push(format!(
                    "cut={}",
                    if self.settings.work_cut {
                        "work"
                    } else {
                        "points"
                    }
                ));
                fields.push(format!("recut={}", if self.recut { "yes" } else { "no" }));
                // The near rule (Phase 6 T12): its crossovers and the pairs converted,
                // summed over the ranks.
                let pairs: [usize; 4] =
                    core::array::from_fn(|k| rows.iter().map(|r| r.near_pairs[k]).sum());
                fields.push(format!("near={}", near_name(self.settings.near_rule)));
                fields.push(format!(
                    "near_crossovers={}",
                    self.near_crossovers
                        .map_or("none".to_owned(), |[w, x]| format!("{w:.1},{x:.1}"))
                ));
                fields.push(format!("w_leaf_pairs={}", pairs[0]));
                fields.push(format!("w_converted={}", pairs[1]));
                fields.push(format!("x_leaf_pairs={}", pairs[2]));
                fields.push(format!("x_converted={}", pairs[3]));
                let w: Vec<String> = (0..W_ENTRY_BINS.len())
                    .map(|k| {
                        rows.iter()
                            .map(|r| r.w_entries[k])
                            .sum::<usize>()
                            .to_string()
                    })
                    .collect();
                fields.push(format!("w_entries={}", w.join("/")));
                fields.push(format!("routing={}", routing_name(self.settings.routing)));
                fields.push(format!(
                    "modelled_max_ms={}",
                    ms(max(&|r: &RankRow| r.modelled))
                ));
                fields.push(format!(
                    "modelled_imbalance={:.3}",
                    imbalance(&|r| r.modelled)
                ));
                fields.push(format!(
                    "sources_imbalance={:.3}",
                    imbalance(&|r| r.sources as f64)
                ));
                fields.push(format!(
                    "leaves_imbalance={:.3}",
                    imbalance(&|r| r.leaves as f64)
                ));
                fields.push(format!(
                    "v_pairs_imbalance={:.3}",
                    imbalance(&|r| r.lists.v as f64)
                ));
                fields.push(format!(
                    "v_pairs_max={}",
                    rows.iter().map(|r| r.lists.v).max().unwrap_or(0)
                ));
                fields.push(format!(
                    "downward_us_per_pair={:.4}",
                    1e6 * path.per_v_pair(rows, |r| path.medians[r][DOWNWARD])
                ));
                if !path.kinds.is_empty() {
                    for (k, key) in [(1, "m2m"), (2, "m2l"), (4, "l2l")] {
                        fields.push(format!("{key}_max_ms={}", ms(path.kind(k)[0])));
                    }
                    fields.push(format!(
                        "m2l_us_per_pair={:.4}",
                        1e6 * path.per_v_pair(rows, |r| path.kinds[r][2])
                    ));
                }
                if let Some(fit) = path.fit(rows) {
                    let alpha: Vec<String> = fit.alpha.iter().map(|a| format!("{a:.4}")).collect();
                    fields.push(format!("alpha_ns={}", alpha.join(",")));
                    fields.push(format!("model_error_max={:.4}", fit.error_max));
                    fields.push(format!("kinds_imbalance={:.3}", fit.measured_imbalance));
                    fields.push(format!("fit_imbalance={:.3}", fit.fitted_imbalance));
                }
                fields.push(format!("compute_max_ms={}", ms(compute[0])));
                fields.push(format!("compute_mean_ms={}", ms(compute[2])));
                fields.push(format!("compute_imbalance={:.3}", compute[0] / compute[2]));
                fields.push(format!("exposed_ms={}", ms(exposed)));
                fields.push(format!("exchanges_ms={}", ms(exchanges)));
                if path.overlapped {
                    // The overlapped path's details (Phase 5S T9).
                    let n = OVERLAP_PARTS.len();
                    let (level, wait) = (0..TOP_TREE_LEVELS)
                        .map(|l| (l, path.overlap_part(n + l)[0]))
                        .fold((0, 0.0), |a, b| if b.1 > a.1 { b } else { a });
                    fields.push(format!(
                        "multipoles_progress_max_ms={}",
                        ms(path.overlap_part(2)[0])
                    ));
                    fields.push(format!(
                        "multipoles_sends_wait_max_ms={}",
                        ms(path.overlap_part(3)[0])
                    ));
                    fields.push(format!("level_wait_max_ms={}", ms(wait)));
                    fields.push(format!("level_wait_max_level={level}"));
                }
                if let Some(e) = &self.errors {
                    fields.push(format!("phi_l2={:.3e}", e.potential.l2));
                    fields.push(format!("grad_l2={:.3e}", e.gradient.l2));
                    fields.push(format!("direct_pairs_max={}", e.direct_pairs));
                    fields.push(format!("direct_ms={}", ms(e.direct_seconds)));
                }
                if let Some(r) = &self.reference {
                    if let Some(strategy) = self.settings.reference_strategy {
                        fields.push(format!("reference_strategy={strategy:?}"));
                    }
                    fields.push(format!("reference_phi_l2={:.2e}", r.potential_l2));
                    fields.push(format!("reference_grad_l2={:.2e}", r.gradient_l2));
                }
                if let Some(d) = self.differing {
                    fields.push(format!("overlap_differing={d}"));
                }
                // The owned top tree (Phase 5S T5), maxima over the ranks (counts and bytes).
                let vb = if self.precision == "f32" { 4.0 } else { 8.0 };
                let top = |key: &str, f: &dyn Fn(&RankRow) -> f64| format!("{key}={}", max(f));
                fields.extend([
                    top("held_h_max", &|r| r.held[..4].iter().sum::<usize>() as f64),
                    top("held_outside_h_max", &|r| r.held[4] as f64),
                    top("global_m2m_max", &|r| r.top_tree.global_m2m as f64),
                    top("phase5_global_m2m", &|r| {
                        r.top_tree.phase5_global_m2m as f64
                    }),
                    top("phase5_coarse_bytes_max", &|r| {
                        r.top_tree.phase5_coarse_received as f64 * vb
                    }),
                    top("reduction_sent_max", &|r| {
                        r.traffic.reduction.messages_sent as f64
                    }),
                    top("reduction_received_max", &|r| {
                        r.traffic.reduction.messages_received as f64
                    }),
                    top("reduction_bytes_max", &|r| {
                        r.traffic.reduction.values_received as f64 * vb
                    }),
                    top("reduction_level_messages_max", &|r| {
                        r.top_tree
                            .reduction_messages
                            .iter()
                            .copied()
                            .max()
                            .unwrap_or(0) as f64
                    }),
                    top("reduction_levels_max", &|r| {
                        r.top_tree
                            .reduction_messages
                            .iter()
                            .filter(|&&m| m > 0)
                            .count() as f64
                    }),
                    top("top_bytes_max", &|r| r.traffic.top_received as f64 * vb),
                    top("top_boxes_written_max", &|r| r.traffic.top_boxes[1] as f64),
                    top("multipole_bytes_max", &|r| {
                        r.traffic.multipoles.values_received as f64 * vb
                    }),
                    top("multipole_messages_max", &|r| {
                        r.traffic.multipoles.messages_received as f64
                    }),
                    top("multipole_messages_sent_max", &|r| {
                        r.traffic.multipoles.messages_sent as f64
                    }),
                    top("multipole_neighbours_max", &|r| {
                        r.top_tree.multipole_neighbours as f64
                    }),
                    top("multipole_off_node_neighbours_max", &|r| {
                        r.top_tree.multipole_off_node_neighbours as f64
                    }),
                    top("multipole_neighbour_nodes_max", &|r| {
                        r.top_tree.multipole_neighbour_nodes as f64
                    }),
                    top("multipole_off_node_messages_max", &|r| {
                        r.top_tree.multipole_off_node_messages as f64
                    }),
                    top("multipole_fan_out_max", &|r| {
                        r.top_tree.multipole_fan_out as f64
                    }),
                    top("multipole_depth_max", &|r| {
                        r.top_tree.multipole_depth as f64
                    }),
                    top("multipole_forwarded_max", &|r| {
                        r.top_tree.multipole_forwarded as f64
                    }),
                    top("top_threshold", &|r| r.top_tree.top_threshold as f64),
                ]);
                fields.push(format!(
                    "memory_model_max_mb={}",
                    mb(max(&|r: &RankRow| r.memory.total()))
                ));
                fields.push(format!("resident_max_mb={}", mb(resident)));
                fields.push(format!("resident_input_max_mb={}", mb(resident_input)));
                let placement = &self.placement;
                fields.push(format!("nodes={}", placement.nodes));
                fields.push(format!(
                    "ranks_per_node={}",
                    placement.ranks_per_node_range().1
                ));
                if let Some(job) = &placement.slurm {
                    fields.push(format!("job={}", job.id));
                }
                format!("summary: {}", fields.join(" "))
            })
            .collect()
    }
}

impl fmt::Display for Report {
    /// The report in Markdown: the heading, the settings, the per-rank tables (up to
    /// [`PER_RANK_ROWS`] ranks), one block of stage times per path, the reduced tables,
    /// the checks, and the [`summaries`](Report::summaries).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = &self.settings;
        let value_bytes = if self.precision == "f32" { 4 } else { 8 };
        writeln!(f, "## scaling: {}", self.heading())?;
        writeln!(f)?;
        writeln!(f, "- machine: {}", self.machine)?;
        writeln!(
            f,
            "- ranks {}, threads per rank {}; {}",
            self.ranks, s.threads, self.threading
        )?;
        writeln!(
            f,
            "- problem: N = {}, {}; seed {:#x}, every point generated by index; \
             max_level {MAX_LEVEL}, {MAX_POINTS_PER_LEAF} points per leaf; {} levels",
            self.workload.n,
            self.workload.distribution.description(),
            self.workload.seed,
            self.nlevels
        )?;
        writeln!(
            f,
            "- precision {}, p = {}, strategy {:?}, P2P kernel {}, gradients; input: {}; \
             overlap: {}; kind timings: {}",
            self.precision,
            s.p,
            self.strategy,
            self.p2p,
            s.input.description(),
            s.overlap.name(),
            if s.kinds { "Synchronous" } else { "off" }
        )?;
        writeln!(
            f,
            "- {} build(s), the median per part; {} timed evaluations per path after one \
             warm-up, charge vector 0, the median per rank and stage, then reduced over the \
             ranks",
            s.builds.max(1),
            s.evaluations
        )?;
        let placement = &self.placement;
        if placement.nodes <= 1 {
            writeln!(
                f,
                "- every time is measured on this machine ({} ranks x {} threads, release \
                 build unless the machine line says otherwise), one node over shared memory; \
                 memory is the formula of distributed-fmm.md §9.2 (*model*) and {}",
                self.ranks, s.threads, self.resident_kind
            )?;
        } else {
            writeln!(
                f,
                "- every time is measured on these {} nodes ({} ranks x {} threads, release \
                 build unless the machine line says otherwise), over the network between the \
                 nodes and shared memory within each; memory is the formula of \
                 distributed-fmm.md §9.2 (*model*) and {}",
                placement.nodes, self.ranks, s.threads, self.resident_kind
            )?;
        }
        let (fewest, most) = placement.ranks_per_node_range();
        let mut hosts: Vec<&str> = Vec::new();
        for (r, host) in placement.hosts.iter().enumerate() {
            if placement.node[r] == hosts.len() {
                hosts.push(host);
            }
        }
        writeln!(
            f,
            "- placement: {} node(s) ({}), ranks per node {}, threads per rank {}; MPI \
             library: {}",
            placement.nodes,
            hosts.join(", "),
            if fewest == most {
                format!("{most}")
            } else {
                format!("{fewest}–{most}")
            },
            s.threads,
            placement.mpi_library
        )?;
        if let Some(job) = &placement.slurm {
            writeln!(f, "- Slurm: {job}")?;
        }
        writeln!(
            f,
            "- the cut: {}; the modelled cost's constants: {}",
            if !s.work_cut {
                "the blocks cut by points only"
            } else if self.recut {
                "the blocks cut again by modelled cost (Phase 5S T7)"
            } else {
                "the second cut gained less than 2 % and kept the cut by points"
            },
            self.cost_model
        )?;
        writeln!(
            f,
            "- the multipole exchange: {} (Phase 5S T8s)",
            routing_name(s.routing)
        )?;
        let pairs: [usize; 4] =
            core::array::from_fn(|k| self.rows.iter().map(|r| r.near_pairs[k]).sum());
        let w: Vec<String> = W_ENTRY_BINS
            .iter()
            .enumerate()
            .map(|(k, bin)| {
                format!(
                    "{bin} {}",
                    self.rows.iter().map(|r| r.w_entries[k]).sum::<usize>()
                )
            })
            .collect();
        writeln!(
            f,
            "- W entries of the local leaves, by box and source points (after the near rule, \
             all ranks): {}",
            w.join(", ")
        )?;
        match self.near_crossovers {
            None => writeln!(f, "- the near rule: off (Phase 6 T12)")?,
            Some([w, x]) => writeln!(
                f,
                "- the near rule: {} (Phase 6 T12), N_W {w:.1} and N_X {x:.1} points; W pairs \
                 between leaves {} of which {} converted, X pairs between leaves {} of which \
                 {} converted (all ranks)",
                near_name(s.near_rule),
                pairs[0],
                pairs[1],
                pairs[2],
                pairs[3]
            )?,
        }
        writeln!(f)?;

        let per_rank = self.ranks <= PER_RANK_ROWS;
        if per_rank {
            writeln!(f, "### Work, held boxes and memory per rank")?;
            writeln!(f)?;
            writeln!(
                f,
                "| rank | input | sources | targets | leaves (empty) | near pairs | V pairs | \
                 W pairs | X pairs | held boxes own / halo / shared / other top / only T | \
                 ghost sources | \
                 memory model, MB (coefficients) | resident, MB |"
            )?;
            writeln!(f, "| --- | {} |", ["---"; 12].join(" | "))?;
            for (r, row) in self.rows.iter().enumerate() {
                writeln!(
                    f,
                    "| {r} | {} | {} | {} | {} ({}) | {} | {} | {} | {} | {} / {} / {} / {} / {} \
                     | {} | \
                     {} ({}) | {} |",
                    row.input,
                    row.sources,
                    row.targets,
                    row.leaves,
                    row.empty_leaves,
                    row.lists.u,
                    row.lists.v,
                    row.lists.w,
                    row.lists.x,
                    row.held[0],
                    row.held[1],
                    row.held[2],
                    row.held[3],
                    row.held[4],
                    row.ghost_sources,
                    mb(row.memory.total()),
                    mb(row.memory.coefficients),
                    row.resident.map_or("–".to_string(), mb)
                )?;
            }
            writeln!(f)?;
            writeln!(f, "### Build per rank, ms")?;
            writeln!(f)?;
            writeln!(f, "| rank | {} |", BUILD_PARTS.join(" | "))?;
            writeln!(f, "| --- | {} |", ["---"; NBUILD].join(" | "))?;
            for (r, row) in self.rows.iter().enumerate() {
                let cells: Vec<String> = row.build.iter().map(|&t| ms(t)).collect();
                writeln!(f, "| {r} | {} |", cells.join(" | "))?;
            }
            writeln!(f)?;
            writeln!(f, "### Traffic per rank and evaluation (bytes)")?;
            writeln!(f)?;
            writeln!(
                f,
                "| rank | sources: msgs sent / recv | sources: bytes sent / recv | upward \
                 reduction: msgs sent / recv | upward reduction: bytes sent / recv | \
                 multipoles: msgs sent / recv | multipoles: bytes sent / recv | replicated top: \
                 boxes own / others, bytes own / others | charges forward: bytes sent / recv | \
                 output backward: bytes sent / recv |"
            )?;
            writeln!(f, "| --- | {} |", ["---"; 9].join(" | "))?;
            for (r, row) in self.rows.iter().enumerate() {
                let t = &row.traffic;
                let b = |v: usize| v * value_bytes;
                writeln!(
                    f,
                    "| {r} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {}, \
                     {} / {} | {} / {} | {} / {} |",
                    t.sources.messages_sent,
                    t.sources.messages_received,
                    b(t.sources.values_sent),
                    b(t.sources.values_received),
                    t.reduction.messages_sent,
                    t.reduction.messages_received,
                    b(t.reduction.values_sent),
                    b(t.reduction.values_received),
                    t.multipoles.messages_sent,
                    t.multipoles.messages_received,
                    b(t.multipoles.values_sent),
                    b(t.multipoles.values_received),
                    t.top_boxes[0],
                    t.top_boxes[1],
                    b(t.top_sent),
                    b(t.top_received),
                    b(t.charges[0]),
                    b(t.charges[1]),
                    b(t.output[0]),
                    b(t.output[1])
                )?;
            }
            writeln!(f)?;
            writeln!(f, "### Placement per rank")?;
            writeln!(f)?;
            writeln!(f, "| rank | node | host | CPUs (Cpus_allowed_list) |")?;
            writeln!(f, "| --- | --- | --- | --- |")?;
            for r in 0..self.ranks.min(placement.hosts.len()) {
                writeln!(
                    f,
                    "| {r} | {} | {} | {} |",
                    placement.node[r], placement.hosts[r], placement.cpus[r]
                )?;
            }
            writeln!(f)?;
        } else {
            writeln!(
                f,
                "Per-rank tables left out above {PER_RANK_ROWS} ranks; the reduced tables \
                 follow."
            )?;
            writeln!(f)?;
        }

        // The work, the build, the traffic and the memory reduced over the ranks.
        writeln!(
            f,
            "### Reduced over the ranks: work, build, traffic, memory"
        )?;
        writeln!(f)?;
        writeln!(f, "| quantity | max | min | mean | max / mean |")?;
        writeln!(f, "| --- | --- | --- | --- | --- |")?;
        let rows = &self.rows;
        let line = |f: &mut fmt::Formatter<'_>,
                    name: &str,
                    get: &dyn Fn(&RankRow) -> f64,
                    show: &dyn Fn(f64) -> String|
         -> fmt::Result {
            let [max, min, mean] = reduce_over(rows.iter().map(get));
            let ratio = if mean > 0.0 {
                format!("{:.3}", max / mean)
            } else {
                "–".to_string()
            };
            writeln!(
                f,
                "| {name} | {} | {} | {} | {ratio} |",
                show(max),
                show(min),
                show(mean)
            )
        };
        let count = |v: f64| format!("{v:.0}");
        let bytes = |v: f64| format!("{v:.0}");
        line(f, "sources owned", &|r| r.sources as f64, &count)?;
        line(
            f,
            "modelled cost (the run's constants), ms",
            &|r| r.modelled,
            &|v| ms(v),
        )?;
        for (k, kind) in KINDS.iter().enumerate() {
            line(
                f,
                &format!("work: {kind} count"),
                &|r| r.kind_counts[k] as f64,
                &count,
            )?;
        }
        line(f, "leaves", &|r| r.leaves as f64, &count)?;
        line(f, "near pairs", &|r| r.lists.u as f64, &count)?;
        line(f, "V pairs", &|r| r.lists.v as f64, &count)?;
        line(f, "W pairs", &|r| r.lists.w as f64, &count)?;
        line(f, "X pairs", &|r| r.lists.x as f64, &count)?;
        line(
            f,
            "held boxes",
            &|r| r.held.iter().sum::<usize>() as f64,
            &count,
        )?;
        line(
            f,
            "held boxes of H, the locally essential set",
            &|r| r.held[..4].iter().sum::<usize>() as f64,
            &count,
        )?;
        for (k, part) in HELD_PARTS.iter().enumerate() {
            line(
                f,
                &format!("held boxes: {part}"),
                &|r| r.held[k] as f64,
                &count,
            )?;
        }
        line(
            f,
            "held boxes not own (fraction)",
            &|r| {
                (r.held.iter().sum::<usize>() - r.held[0]) as f64
                    / r.held.iter().sum::<usize>().max(1) as f64
            },
            &|v| format!("{v:.3}"),
        )?;
        line(f, "ghost sources", &|r| r.ghost_sources as f64, &count)?;
        line(
            f,
            "plan arrays (box index, leaves, views), MB",
            &|r| r.plan_bytes as f64,
            &|v| mb(v),
        )?;
        for (k, part) in BUILD_PARTS.iter().enumerate() {
            line(f, &format!("build: {part}, ms"), &|r| r.build[k], &|v| {
                ms(v)
            })?;
        }
        let vb = value_bytes as f64;
        line(
            f,
            "source exchange: messages sent",
            &|r| r.traffic.sources.messages_sent as f64,
            &count,
        )?;
        line(
            f,
            "source exchange: bytes received",
            &|r| r.traffic.sources.values_received as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "multipole exchange: messages sent (all levels)",
            &|r| r.traffic.multipoles.messages_sent as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: bytes received",
            &|r| r.traffic.multipoles.values_received as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "multipole exchange: neighbours (all levels)",
            &|r| r.top_tree.multipole_neighbours as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: neighbours on other nodes",
            &|r| r.top_tree.multipole_off_node_neighbours as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: other nodes received from",
            &|r| r.top_tree.multipole_neighbour_nodes as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: messages sent to other nodes (all levels)",
            &|r| r.top_tree.multipole_off_node_messages as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: most ranks one box is sent to",
            &|r| r.top_tree.multipole_fan_out as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: stages (the forwarding tree's depth)",
            &|r| r.top_tree.multipole_depth as f64,
            &count,
        )?;
        line(
            f,
            "multipole exchange: chunks forwarded (all levels)",
            &|r| r.top_tree.multipole_forwarded as f64,
            &count,
        )?;
        line(
            f,
            "replicated top: readers' threshold",
            &|r| r.top_tree.top_threshold as f64,
            &count,
        )?;
        line(
            f,
            "upward reduction: messages sent (all levels)",
            &|r| r.traffic.reduction.messages_sent as f64,
            &count,
        )?;
        line(
            f,
            "upward reduction: messages received (all levels)",
            &|r| r.traffic.reduction.messages_received as f64,
            &count,
        )?;
        line(
            f,
            "upward reduction: bytes received",
            &|r| r.traffic.reduction.values_received as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "replicated top: boxes held and written",
            &|r| r.traffic.top_boxes[1] as f64,
            &count,
        )?;
        line(
            f,
            "replicated top: bytes received (the all-gather)",
            &|r| r.traffic.top_received as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "global M2M pairs (owned `Global` boxes)",
            &|r| r.top_tree.global_m2m as f64,
            &count,
        )?;
        line(
            f,
            "Phase 5: global M2M pairs (every `Global` box)",
            &|r| r.top_tree.phase5_global_m2m as f64,
            &count,
        )?;
        line(
            f,
            "Phase 5: coarse gather, bytes received",
            &|r| r.top_tree.phase5_coarse_received as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "charges forward: bytes sent",
            &|r| r.traffic.charges[0] as f64 * vb,
            &bytes,
        )?;
        line(
            f,
            "output backward: bytes sent",
            &|r| r.traffic.output[0] as f64 * vb,
            &bytes,
        )?;
        line(f, "memory model, MB", &|r| r.memory.total(), &|v| mb(v))?;
        line(
            f,
            "memory model: coefficients, MB",
            &|r| r.memory.coefficients,
            &|v| mb(v),
        )?;
        line(
            f,
            "memory model: replicated top buffers and keys, MB",
            &|r| (r.traffic.top_sent + r.traffic.top_received) as f64 * vb + r.memory.top,
            &|v| mb(v),
        )?;
        if rows.iter().any(|r| r.memory.plane_waves > 0.0) {
            line(
                f,
                "memory model: plane waves (one frame of the largest level, the threads' \
                 scratch), MB",
                &|r| r.memory.plane_waves,
                &|v| mb(v),
            )?;
        }
        if rows.iter().all(|r| r.resident.is_some()) {
            line(
                f,
                "resident, MB",
                &|r| r.resident.unwrap_or(f64::NAN),
                &|v| mb(v),
            )?;
        }
        if rows.iter().all(|r| r.resident_input.is_some()) {
            line(
                f,
                "resident after the input (the harness, before the builds), MB",
                &|r| r.resident_input.unwrap_or(f64::NAN),
                &|v| mb(v),
            )?;
        }
        writeln!(f)?;

        // The upward reduction per level (Phase 5S T5): only the levels with a message.
        let levels: Vec<usize> = (0..TOP_TREE_LEVELS)
            .filter(|&l| rows.iter().any(|r| r.top_tree.reduction_messages[l] > 0))
            .collect();
        if !levels.is_empty() {
            writeln!(f, "### Upward reduction per level")?;
            writeln!(f)?;
            writeln!(
                f,
                "| level | ranks receiving | messages received: max, sum | bytes received: max, \
                 sum |"
            )?;
            writeln!(f, "| --- | --- | --- | --- |")?;
            for l in levels {
                let messages = rows.iter().map(|r| r.top_tree.reduction_messages[l]);
                let values = rows.iter().map(|r| r.top_tree.reduction_values[l]);
                writeln!(
                    f,
                    "| {l} | {} | {}, {} | {}, {} |",
                    messages.clone().filter(|&m| m > 0).count(),
                    messages.clone().max().unwrap_or(0),
                    messages.sum::<usize>(),
                    values.clone().max().unwrap_or(0) * value_bytes,
                    values.sum::<usize>() * value_bytes
                )?;
            }
            writeln!(f)?;
        }

        for path in &self.paths {
            let name = if path.overlapped {
                "overlapped"
            } else {
                "blocking"
            };
            writeln!(
                f,
                "### Evaluation, {name}: {}, overlap {}",
                self.heading(),
                if path.overlapped { "on" } else { "off" }
            )?;
            writeln!(f)?;
            writeln!(
                f,
                "Evaluation wall time (median over evaluations of the max over ranks of the \
                 total): {} ms",
                ms(path.wall)
            )?;
            writeln!(f)?;
            if per_rank {
                writeln!(f, "| rank | {} |", STAGES.join(" | "))?;
                writeln!(f, "| --- | {} |", ["---"; NSTAGES].join(" | "))?;
                for (r, m) in path.medians.iter().enumerate() {
                    let cells: Vec<String> = m.iter().map(|&t| ms(t)).collect();
                    writeln!(f, "| {r} | {} |", cells.join(" | "))?;
                }
                writeln!(f)?;
            }
            writeln!(
                f,
                "| stage, ms (median per rank) | max | min | mean | max / mean |"
            )?;
            writeln!(f, "| --- | --- | --- | --- | --- |")?;
            for (k, stage) in STAGES.iter().enumerate() {
                let [max, min, mean] = path.reduced(k);
                writeln!(
                    f,
                    "| {stage} | {} | {} | {} | {} |",
                    ms(max),
                    ms(min),
                    ms(mean),
                    if mean.is_nan() || mean == 0.0 {
                        "–".to_string()
                    } else {
                        format!("{:.3}", max / mean)
                    }
                )?;
            }
            let [max, min, mean] = path.compute();
            writeln!(
                f,
                "| compute (upward local + downward + leaves) | {} | {} | {} | {:.3} |",
                ms(max),
                ms(min),
                ms(mean),
                max / mean
            )?;
            writeln!(f)?;
            if !path.kinds.is_empty() {
                writeln!(
                    f,
                    "| kind, ms (median per rank; `KindTiming::Synchronous`) | max | min | mean \
                     | max / mean |"
                )?;
                writeln!(f, "| --- | --- | --- | --- | --- |")?;
                for (k, kind) in KINDS.iter().enumerate() {
                    let [max, min, mean] = path.kind(k);
                    writeln!(
                        f,
                        "| {kind} | {} | {} | {} | {} |",
                        ms(max),
                        ms(min),
                        ms(mean),
                        if mean > 0.0 {
                            format!("{:.3}", max / mean)
                        } else {
                            "–".to_string()
                        }
                    )?;
                }
                writeln!(f)?;
                writeln!(
                    f,
                    "Per V pair, of the slowest rank in each: M2L {:.4} µs, downward {:.4} µs",
                    1e6 * path.per_v_pair(rows, |r| path.kinds[r][2]),
                    1e6 * path.per_v_pair(rows, |r| path.medians[r][DOWNWARD])
                )?;
                writeln!(f)?;
                if let Some(fit) = path.fit(rows) {
                    let alpha: Vec<String> = KINDS
                        .iter()
                        .zip(fit.alpha)
                        .map(|(kind, a)| format!("{kind} {a:.4}"))
                        .collect();
                    writeln!(
                        f,
                        "The cost model (Phase 5S T7): constants fitted to these kind times, ns \
                         per counted unit: {}. With them the model is within {:.2} % of every \
                         rank's sum of kind times; max / mean of that sum {:.3}, of the model \
                         with the fitted constants {:.3}, with the run's constants {:.3}.",
                        alpha.join(", "),
                        100.0 * fit.error_max,
                        fit.measured_imbalance,
                        fit.fitted_imbalance,
                        fit.modelled_imbalance
                    )?;
                    writeln!(f)?;
                }
            }
            // The slowest rank in compute, and its share of the work.
            let slowest = path
                .medians
                .iter()
                .enumerate()
                .map(|(r, m)| (r, COMPUTE.iter().map(|&k| m[k]).sum::<f64>()))
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
                .0;
            let row = &rows[slowest];
            let mean_of = |get: &dyn Fn(&RankRow) -> f64| {
                rows.iter().map(get).sum::<f64>() / rows.len() as f64
            };
            writeln!(
                f,
                "Slowest rank in compute: {slowest}; its work over the mean: sources {:.3}, \
                 leaves {:.3}, near pairs {:.3}, V pairs {:.3}, W pairs {:.3}, X pairs {:.3}",
                row.sources as f64 / mean_of(&|r| r.sources as f64),
                row.leaves as f64 / mean_of(&|r| r.leaves as f64),
                row.lists.u as f64 / mean_of(&|r| r.lists.u as f64),
                row.lists.v as f64 / mean_of(&|r| r.lists.v as f64),
                row.lists.w as f64 / mean_of(&|r| r.lists.w as f64).max(f64::MIN_POSITIVE),
                row.lists.x as f64 / mean_of(&|r| r.lists.x as f64).max(f64::MIN_POSITIVE),
            )?;
            writeln!(f)?;
        }
        if let [blocking, overlapped] = &self.paths[..] {
            let (exposed, _) = overlapped.exchanges();
            let (_, exchanges) = blocking.exchanges();
            // The upward reduction beside the criterion (Phase 5S T6), as the coarse gather
            // was in Phase 5: each level's send to the end of its wait.
            let reduction = |path: &PathTimes| path.reduced(7)[0];
            writeln!(
                f,
                "C5.2 measure (max over ranks): exposed waits {} ms against blocking \
                 exchanges {} ms = {:.1}%; evaluation {} ms blocking, {} ms overlapped; {} \
                 values of the two paths differ in any bit; the upward reduction (not in the \
                 criterion) {} ms overlapped, {} ms blocking",
                ms(exposed),
                ms(exchanges),
                100.0 * exposed / exchanges,
                ms(blocking.wall),
                ms(overlapped.wall),
                self.differing.unwrap_or(0),
                ms(reduction(overlapped)),
                ms(reduction(blocking))
            )?;
            writeln!(f)?;
            // The overlapped path's details (Phase 5S T9): the test calls, the final wait
            // for the sends and the wait for each level of the multipole exchange.
            writeln!(
                f,
                "Overlapped path, the exchanges' test calls and the multipole exchange's waits \
                 (ms; per rank the median over the evaluations):"
            )?;
            writeln!(f)?;
            writeln!(f, "| part | max | min | mean |")?;
            writeln!(f, "| --- | ---: | ---: | ---: |")?;
            let parts = OVERLAP_PARTS.iter().map(|name| (*name).to_string()).chain(
                (0..self.nlevels.min(TOP_TREE_LEVELS))
                    .map(|l| format!("multipoles: wait for level {l}")),
            );
            for (k, name) in parts.enumerate() {
                let [max, min, mean] = overlapped.overlap_part(k);
                writeln!(f, "| {name} | {} | {} | {} |", ms(max), ms(min), ms(mean))?;
            }
            writeln!(f)?;
        }
        match &self.errors {
            Some(e) => writeln!(
                f,
                "Errors against the direct sum ({} charge vectors, {} sampled targets, root \
                 mean squares): φ L2 {:.3e}, max {:.3e}; ∇φ L2 {:.3e}, max {:.3e}. The \
                 direct sum: each rank over its own sources (Neumaier-compensated in f64), \
                 one all-reduce; at most {} pairs per rank and vector, {} ms on the slowest \
                 rank over the vectors",
                e.vectors,
                e.sampled,
                e.potential.l2,
                e.potential.max,
                e.gradient.l2,
                e.gradient.max,
                e.direct_pairs,
                ms(e.direct_seconds)
            )?,
            None => writeln!(f, "Errors against the direct sum: not checked in this run")?,
        }
        match &self.reference {
            Some(r) => writeln!(
                f,
                "Against the one-rank FMM{} over every point in index order (charge vector \
                 0, every target): relative L2 of φ {:.2e}, of ∇φ {:.2e}; max relative \
                 {:.2e}; {} values differ in any bit",
                s.reference_strategy
                    .map_or(String::new(), |strategy| format!(" with {strategy:?}")),
                r.potential_l2,
                r.gradient_l2,
                r.max_relative,
                r.differing
            )?,
            None => writeln!(f, "Against the one-rank FMM: not checked in this run")?,
        }
        writeln!(f)?;
        for line in self.summaries() {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

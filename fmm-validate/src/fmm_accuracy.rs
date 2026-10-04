//! The accuracy of the complete FMM on a uniform tree (C3.2) and on adaptive trees of
//! clustered points (C3.3): the core of the `fmm_accuracy` example.
//!
//! The problem ([`Problem`]): N points of a [`Distribution`], sources equal to targets,
//! and several charge vectors uniform in [−1, 1). [`run`] builds an
//! `nd_fmm_exec::fmm::Fmm` with gradients at degree p, with a given number of threads,
//! P2P kernel and backend ([`Execution`]), evaluates every charge vector and measures the
//! output at a fixed sample of the targets against the [`Oracle`]. The output, and so
//! every error, is bit-identical for every number of threads (C3.5), and depends on the
//! P2P kernel only in the last bits (C3S.5); the timings change with both. On a device
//! backend (feature `gpu`, Phase 4 T11) every operator kind runs on the device and the
//! output agrees with the host's within the FMM bounds of docs/phase4/README.md; the
//! [`Run`] then also says what ran where ([`DeviceRun`]).
//!
//! Two kinds of tree ([`Config`]):
//! - [`Config::C32`]: points uniform in the cube [−1, 1)³ and a uniform tree,
//!   `max_level` 4 and one point per leaf as the refinement target, so with enough
//!   points every leaf lies on level 4. Its W and X lists are empty.
//! - [`Config::c33`]: the sphere surface, the Plummer sphere or the Gaussian clusters
//!   (design §8.2), `max_level` 16 and a fixed refinement target of 64 points per leaf,
//!   the same for every distribution. The trees are adaptive, with leaves on many levels
//!   and non-empty W and X lists, which the C3.3 gate needs.
//!
//! # Error measure
//!
//! Per charge vector k, the relative L2 and max errors eₖ of φ and of ∇φ (the Euclidean
//! norm per target) at the sampled targets ([`metrics::ErrorNorms`]), against
//! `direct_sum` in f64 over all sources, divided by 4π: the oracle expands 1/|x − y|,
//! the FMM returns Σ q / (4π |x − y|). A [`Run`] reports the root mean square
//! √((1/K) Σₖ eₖ²) of each, and the range of the φ L2 error over the vectors. Each
//! vector counts equally (docs/phase3/README.md, "Error measures"): with mixed-sign
//! charges the far field of the coarsest V-list level partly cancels at each target
//! by an amount that depends on the charges, so one vector alone measures the draw as
//! much as the FMM.
//!
//! The C3.2 gate compares the φ L2 error with twice [`PREDICTION`], the single-
//! translation error (design §7, as re-derived for p = 18 in T9). The C3.3 gate
//! compares the φ L2 error of each clustered distribution with twice the error of the
//! uniform tree of [`Config::C32`] at the same p and N. f32 runs use charges rounded to
//! f32 and an oracle on the rounded charges, so the input rounding is not counted.
//!
//! [`metrics::ErrorNorms`]: crate::metrics::ErrorNorms

use std::time::Duration;

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{
    Backend, BuildTimings, DeviceStage, DeviceStageTimings, Fmm, FmmBuilder, ListSizes,
    OperatorKind, StageTimings,
};
use nd_fmm_exec::operator::{P2pChoice, SimdScalar};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_exec::threading::ThreadingReport;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;

use crate::metrics::{ErrorAccumulator, ErrorNorms};
use crate::{SplitMix64, points};

/// A point distribution of the accuracy runs (design §8.2), each in [−1, 1]³.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Distribution {
    /// Uniform in the cube [−1, 1)³ ([`points::cube`]).
    Cube,
    /// Uniform on the unit sphere about the origin ([`points::sphere`]).
    Sphere,
    /// The Plummer sphere of scale [`PLUMMER_SCALE`] about the origin, truncated at
    /// radius 1 ([`points::plummer`]).
    Plummer,
    /// Gaussian clusters of width [`CLUSTER_WIDTH`] at [`CLUSTER_CENTRES`], truncated at
    /// 4 σ ([`points::gaussian_clusters`]).
    Clusters,
}

/// The scale a of the Plummer distribution: truncated at 10 a = 1, so the points lie
/// in the unit ball, half of them within 1.29 a.
pub const PLUMMER_SCALE: f64 = 0.1;

/// The width σ of each Gaussian cluster: tight, truncated at 4 σ = 0.08.
pub const CLUSTER_WIDTH: f64 = 0.02;

/// The centres of the five Gaussian clusters, spread irregularly over [−1, 1]³; the
/// second and the fifth lie 0.21 apart, close enough for their trees to meet.
pub const CLUSTER_CENTRES: [[f64; 3]; 5] = [
    [-0.5, -0.5, -0.5],
    [0.5, -0.4, 0.3],
    [-0.3, 0.6, 0.2],
    [0.4, 0.5, -0.6],
    [0.6, -0.3, 0.45],
];

impl Distribution {
    /// Every distribution, the uniform cube first.
    pub const ALL: [Self; 4] = [Self::Cube, Self::Sphere, Self::Plummer, Self::Clusters];

    /// The name of the distribution: "cube", "sphere", "plummer" or "clusters".
    pub fn name(self) -> &'static str {
        match self {
            Self::Cube => "cube",
            Self::Sphere => "sphere",
            Self::Plummer => "plummer",
            Self::Clusters => "clusters",
        }
    }

    /// The distribution of a [`name`](Self::name), if there is one.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == name)
    }

    /// A description of the distribution and its parameters, for reports.
    pub fn description(self) -> String {
        match self {
            Self::Cube => "uniform in the cube [-1, 1)^3".into(),
            Self::Sphere => "uniform on the unit sphere about the origin".into(),
            Self::Plummer => format!(
                "a Plummer sphere about the origin, scale a = {PLUMMER_SCALE}, truncated at \
                 {} a",
                points::PLUMMER_TRUNCATION
            ),
            Self::Clusters => format!(
                "{} Gaussian clusters of width {CLUSTER_WIDTH} at {CLUSTER_CENTRES:?}, \
                 truncated at {} σ, the points spread evenly over them",
                CLUSTER_CENTRES.len(),
                points::GAUSSIAN_TRUNCATION
            ),
        }
    }

    /// Draws `n` points of the distribution from `rng`.
    pub fn points(self, rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
        match self {
            Self::Cube => points::cube(rng, n, [0.0; 3], 1.0),
            Self::Sphere => points::sphere(rng, n, [0.0; 3], 1.0),
            Self::Plummer => points::plummer(rng, n, [0.0; 3], PLUMMER_SCALE),
            Self::Clusters => points::gaussian_clusters(rng, n, &CLUSTER_CENTRES, CLUSTER_WIDTH),
        }
    }
}

/// The single-translation prediction of the relative L2 error of φ, P2M → M2L → L2P
/// pooled over the 316 V-list offsets (design §7, "Single-translation accuracy";
/// docs/phase3/README.md, "Predictions").
///
/// p = 3 and 8 are the Phase 1 T7 values (one draw of 1,000 sources, 1,000 targets per
/// offset). p = 18 is the T9 re-derivation, the median over 33 draws with 10⁴
/// targets per offset; Phase 1's 2.71e-9 had too few targets for the heavy-tailed
/// error of the face offsets.
pub const PREDICTION: [(usize, f64); 3] = [(3, 1.77e-3), (8, 1.08e-5), (18, 6.74e-9)];

/// Returns the prediction at degree `p`, if [`PREDICTION`] has one.
pub fn prediction(p: usize) -> Option<f64> {
    PREDICTION.iter().find(|&&(q, _)| q == p).map(|&(_, e)| e)
}

/// The problem and its tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The point distribution.
    pub distribution: Distribution,
    /// The number of points, sources and targets alike.
    pub n: usize,
    /// The deepest level of a leaf (`FmmBuilder::max_level`).
    pub max_level: usize,
    /// The refinement target (`FmmBuilder::max_points_per_leaf`).
    pub max_points_per_leaf: usize,
    /// The number of targets the error is measured at.
    pub sampled: usize,
    /// The number of charge vectors.
    pub charge_vectors: usize,
    /// The seed of the points and the sample; charge vector k uses `seed + 1 + k`.
    pub seed: u64,
}

impl Config {
    /// The C3.2 problem: N = 10⁵, a uniform level-4 tree, 1,000 sampled targets, eight
    /// charge vectors. The same points, sample and charges as the ignored gate test of
    /// nd-fmm-exec (`tests/accuracy.rs`).
    pub const C32: Self = Self {
        distribution: Distribution::Cube,
        n: 100_000,
        max_level: 4,
        max_points_per_leaf: 1,
        sampled: 1000,
        charge_vectors: 8,
        seed: 0xc32,
    };

    /// The C3.3 problem of `distribution`: N = 10⁵, an adaptive tree with `max_level` 16
    /// and 64 points per leaf as the refinement target, 1,000 sampled targets, eight
    /// charge vectors, seed `0xc33`. The settings are the same for every distribution.
    pub const fn c33(distribution: Distribution) -> Self {
        Self {
            distribution,
            n: 100_000,
            max_level: 16,
            max_points_per_leaf: 64,
            sampled: 1000,
            charge_vectors: 8,
            seed: 0xc33,
        }
    }
}

/// How the FMM of a [`run`] executes: its threads, its P2P kernel and its backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Execution {
    /// The number of rayon threads (`FmmBuilder::threads`), at least 1. With
    /// `Backend::Cpu` they cap the CPU runtime's units per cube instead.
    pub threads: usize,
    /// The P2P kernel of the host path and of host-fallback calls
    /// (`FmmBuilder::p2p_kernel`).
    pub p2p: P2pChoice,
    /// Where the operators run (`FmmBuilder::backend`): the host, or a device backend
    /// compiled in (feature `gpu`) with every kind on the device.
    pub backend: Backend,
}

impl Execution {
    /// `threads` threads, the default P2P kernel, `P2pChoice::Auto`, and the host.
    pub const fn threads(threads: usize) -> Self {
        Self {
            threads,
            p2p: P2pChoice::Auto,
            backend: Backend::Host,
        }
    }
}

/// What a [`run`] on a device backend ran where, fixed at build, and what its last
/// evaluation moved (`Fmm::device_report`, `Fmm::device_counters`; feature `gpu`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRun {
    /// The device: backend, compiler, name, CubeCL version and precisions.
    pub device: String,
    /// The operator kinds on the device, in `OperatorKind::ALL` order.
    pub on_device: Vec<OperatorKind>,
    /// The M2L strategy as the device runs it (`DeviceReport::strategy_name`).
    pub strategy: String,
    /// The GEMMs of the device translations and the rotation M2L: for each family its
    /// level calls, chunks and GEMM.
    pub gemms: String,
    /// How the stages are timed (`DeviceReport::stage_timing`).
    pub stage_timing: String,
    /// The last evaluation's uploads, upload bytes, downloads, download bytes, launches,
    /// syncs and timing windows.
    pub evaluation: [u64; 7],
}

impl DeviceRun {
    /// What `fmm` runs on its device; `None` on the host.
    #[cfg(feature = "gpu")]
    fn of<T: Stored + SimdScalar + Equivalence + Default>(fmm: &Fmm<'_, T>) -> Option<Self> {
        use nd_fmm_exec::device::Gemm;
        use nd_fmm_exec::fmm::Placement;
        let (report, counters) = (fmm.device_report()?, fmm.device_counters()?);
        let mut gemms = Vec::new();
        for (name, kind) in [
            ("M2M", OperatorKind::M2m),
            ("L2L", OperatorKind::L2l),
            ("M2L", OperatorKind::M2l),
        ] {
            let calls: Vec<_> = report.translations_of(kind).collect();
            if calls.is_empty() {
                continue;
            }
            let library = calls.iter().filter(|t| t.gemm == Gemm::Library).count();
            let chunks: usize = calls.iter().map(|t| t.chunks).sum();
            gemms.push(format!(
                "{name} {} calls in {chunks} chunks ({library} library, {} hand-written {})",
                calls.len(),
                calls.len() - library,
                report.gemm_layout
            ));
        }
        if !report.rotations.is_empty() {
            gemms.push(format!(
                "M2L rotation {} calls ({})",
                report.rotations.len(),
                report.rotation_layout
            ));
        }
        let c = counters.evaluation;
        Some(Self {
            device: report.info.to_string(),
            on_device: OperatorKind::ALL
                .into_iter()
                .filter(|&k| report.placement(k) == Placement::Device)
                .collect(),
            strategy: report.strategy_name(),
            gemms: gemms.join("; "),
            stage_timing: report.stage_timing.to_string(),
            evaluation: [
                c.uploads,
                c.upload_bytes,
                c.downloads,
                c.download_bytes,
                c.launches,
                c.syncs,
                c.windows,
            ],
        })
    }

    /// Without the `gpu` feature every run is on the host.
    #[cfg(not(feature = "gpu"))]
    fn of<T: Stored + SimdScalar + Equivalence + Default>(_fmm: &Fmm<'_, T>) -> Option<Self> {
        None
    }
}

/// The points, the sampled targets and the charge vectors of a [`Config`].
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The points of the distribution.
    pub points: Vec<[f64; 3]>,
    /// The positions in `points` of the sampled targets, distinct.
    pub sample: Vec<usize>,
    /// The charge vectors, uniform in [−1, 1).
    pub charges: Vec<Vec<f64>>,
}

impl Problem {
    /// Draws the problem: the points of the distribution from `seed`, then the sample
    /// from the same
    /// generator (a partial Fisher–Yates shuffle), and charge vector k from
    /// `seed + 1 + k`.
    ///
    /// # Panics
    ///
    /// If `config.sampled > config.n`.
    pub fn new(config: &Config) -> Self {
        assert!(config.sampled <= config.n, "more samples than points");
        let mut rng = SplitMix64::new(config.seed);
        let points = config.distribution.points(&mut rng, config.n);
        let mut order: Vec<usize> = (0..config.n).collect();
        for i in 0..config.sampled {
            let j = i + below(&mut rng, config.n - i);
            order.swap(i, j);
        }
        order.truncate(config.sampled);
        let charges = (0..config.charge_vectors as u64)
            .map(|k| points::charges(&mut SplitMix64::new(config.seed + 1 + k), config.n))
            .collect();
        Self {
            points,
            sample: order,
            charges,
        }
    }

    /// The sampled targets.
    pub fn sampled_points(&self) -> Vec<[f64; 3]> {
        self.sample.iter().map(|&i| self.points[i]).collect()
    }

    /// The charge vectors rounded to `T`.
    pub fn charges_as<T: RealScalar>(&self) -> Vec<Vec<T>> {
        self.charges
            .iter()
            .map(|q| q.iter().map(|&v| T::from_f64(v)).collect())
            .collect()
    }
}

/// Uniform in 0..n, from the top 32 bits (n ≤ 2³², with a negligible bias).
fn below(rng: &mut SplitMix64, n: usize) -> usize {
    (((rng.next_u64() >> 32) * n as u64) >> 32) as usize
}

/// The exact potentials and gradients at the sampled targets, per charge vector:
/// `direct_sum` in f64 over all points, divided by 4π.
#[derive(Clone, Debug, PartialEq)]
pub struct Oracle {
    /// φ per charge vector and sampled target.
    pub potential: Vec<Vec<f64>>,
    /// ∇φ per charge vector and sampled target.
    pub gradient: Vec<Vec<[f64; 3]>>,
}

impl Oracle {
    /// The oracle of `problem` for the charge vectors `charges` (the problem's own, or
    /// their rounded values in f64). O(K N n_sampled).
    pub fn new(problem: &Problem, charges: &[Vec<f64>]) -> Self {
        let targets = problem.sampled_points();
        let scale = 4.0 * std::f64::consts::PI;
        let (potential, gradient) = charges
            .iter()
            .map(|q| {
                let mut phi = vec![0.0; targets.len()];
                let mut grad = vec![[0.0; 3]; targets.len()];
                direct_sum(&problem.points, q, &targets, &mut phi, Some(&mut grad));
                (
                    phi.iter().map(|v| v / scale).collect(),
                    grad.iter().map(|g| g.map(|gk| gk / scale)).collect(),
                )
            })
            .unzip();
        Self {
            potential,
            gradient,
        }
    }
}

/// The result of one [`run`].
#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    /// The point distribution.
    pub distribution: Distribution,
    /// "f32" or "f64".
    pub precision: &'static str,
    /// The degree.
    pub p: usize,
    /// The resolved M2L strategy.
    pub strategy: M2lStrategy,
    /// Root mean squares over the charge vectors of the relative L2 and max errors of
    /// φ.
    pub potential: ErrorNorms,
    /// Root mean squares over the charge vectors of the relative L2 and max errors of
    /// ∇φ.
    pub gradient: ErrorNorms,
    /// The smallest and the largest φ L2 error of one charge vector.
    pub potential_l2_range: (f64, f64),
    /// The number of levels.
    pub nlevels: usize,
    /// The number of leaves.
    pub nleaves: usize,
    /// The number of leaves on each level 0, 1, …, `nlevels` − 1.
    pub leaf_levels: Vec<usize>,
    /// Points per leaf, empty leaves included: the smallest, the mean and the largest
    /// count.
    pub points_per_leaf: (usize, f64, usize),
    /// The sizes of the interaction lists.
    pub lists: ListSizes,
    /// The wall time of the build.
    pub build: BuildTimings,
    /// The wall time of each stage, the mean over the charge vectors.
    pub stages: StageTimings,
    /// The rayon threads, the MPI threading level and the BLAS thread variables.
    pub threading: ThreadingReport,
    /// The P2P kernel as it ran (`Fmm::p2p_kernel`): `Reference` or an ISA, never
    /// `Auto`.
    pub p2p: P2pChoice,
    /// The backend.
    pub backend: Backend,
    /// On a device backend, what ran where and what an evaluation moved; `None` on the
    /// host.
    pub device: Option<DeviceRun>,
}

impl Run {
    /// The φ L2 error over the prediction at this p, if there is one.
    pub fn ratio(&self) -> Option<f64> {
        prediction(self.p).map(|e| self.potential.l2 / e)
    }
}

/// Builds the FMM of `problem` in precision `T` at degree `p` with `execution`'s threads,
/// P2P kernel and backend (default strategy, gradients on), evaluates the charge vectors
/// `charges` and measures the output against `oracle` (module documentation).
///
/// `charges` and `oracle` must belong together: the problem's charges for f64, and
/// for f32 the rounded charges with the oracle of their f64 values.
///
/// # Collective operation
///
/// On `comm`, which must have one rank: the points are not redistributed (C5.1).
///
/// # Panics
///
/// If the FMM does not build or evaluate, for example on several ranks, with
/// `threads` > 1 when MPI provides less than `Threading::Funneled`, with a P2P
/// kernel on an ISA this machine cannot run, or on a device backend that is not compiled
/// in or does no arithmetic in `T` (f64 on Metal).
pub fn run<T: Stored + SimdScalar + Equivalence + Default>(
    config: &Config,
    problem: &Problem,
    charges: &[Vec<T>],
    oracle: &Oracle,
    (p, execution): (usize, Execution),
    comm: &SimpleCommunicator,
) -> Run {
    let builder = FmmBuilder::<T>::new(p)
        .max_level(config.max_level)
        .max_points_per_leaf(config.max_points_per_leaf)
        .gradients(true)
        .threads(execution.threads)
        .p2p_kernel(execution.p2p)
        .backend(execution.backend);
    let mut fmm: Fmm<'_, T> = builder
        .build(&problem.points, &problem.points, comm)
        .unwrap_or_else(|error| panic!("the FMM does not build: {error}"));
    let counts = fmm.source_counts();
    let points_per_leaf = (
        counts.iter().copied().min().unwrap_or(0),
        counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64,
        counts.iter().copied().max().unwrap_or(0),
    );
    let leaves = fmm.plan().index().leaves();
    let mut leaf_levels = vec![0; fmm.nlevels()];
    for j in 0..fmm.nleaves() {
        leaf_levels[leaves.level(j)] += 1;
    }
    let mut errors = Vec::with_capacity(charges.len());
    let mut stages = Vec::with_capacity(charges.len());
    for ((q, phi), grad) in charges.iter().zip(&oracle.potential).zip(&oracle.gradient) {
        let output = fmm
            .evaluate(q)
            .unwrap_or_else(|error| panic!("the FMM does not evaluate: {error}"));
        let gradient = output.gradient.as_ref().expect("built with gradients");
        let mut potential_errors = ErrorAccumulator::new();
        let mut gradient_errors = ErrorAccumulator::new();
        for (r, &i) in problem.sample.iter().enumerate() {
            potential_errors.add_values(&output.potential[i..=i], &phi[r..=r]);
            gradient_errors.add_vectors(&gradient[i..=i], &grad[r..=r]);
        }
        errors.push((potential_errors.finish(), gradient_errors.finish()));
        stages.push(output.timings);
    }
    let rms = |f: &dyn Fn(&(ErrorNorms, ErrorNorms)) -> f64| {
        (errors.iter().map(|e| f(e).powi(2)).sum::<f64>() / errors.len() as f64).sqrt()
    };
    let l2 = errors.iter().map(|e| e.0.l2);
    Run {
        distribution: config.distribution,
        precision: if size_of::<T>() == 4 { "f32" } else { "f64" },
        p,
        strategy: fmm.strategy(),
        potential: ErrorNorms {
            l2: rms(&|e| e.0.l2),
            max: rms(&|e| e.0.max),
        },
        gradient: ErrorNorms {
            l2: rms(&|e| e.1.l2),
            max: rms(&|e| e.1.max),
        },
        potential_l2_range: (
            l2.clone().fold(f64::INFINITY, f64::min),
            l2.fold(0.0, f64::max),
        ),
        nlevels: fmm.nlevels(),
        nleaves: fmm.nleaves(),
        leaf_levels,
        points_per_leaf,
        lists: fmm.list_sizes(),
        build: fmm.build_timings(),
        stages: mean(&stages),
        threading: fmm.threading().clone(),
        p2p: fmm.p2p_kernel(),
        backend: execution.backend,
        device: DeviceRun::of(&fmm),
    }
}

/// The mean of each stage over `timings`.
fn mean(timings: &[StageTimings]) -> StageTimings {
    let n = timings.len().max(1) as u32;
    let average = |f: fn(&StageTimings) -> Duration| -> Duration {
        timings.iter().map(f).sum::<Duration>() / n
    };
    StageTimings {
        load: average(|t| t.load),
        exchange_sources: average(|t| t.exchange_sources),
        upward_local: average(|t| t.upward_local),
        upward_global: average(|t| t.upward_global),
        exchange_multipoles: average(|t| t.exchange_multipoles),
        downward: average(|t| t.downward),
        evaluate_leaves: average(|t| t.evaluate_leaves),
        output: average(|t| t.output),
        device: mean_device(timings),
    }
}

/// The mean of each device stage over `timings`, if every evaluation has device times.
fn mean_device(timings: &[StageTimings]) -> Option<DeviceStageTimings> {
    let device: Vec<DeviceStageTimings> =
        timings.iter().map(|t| t.device).collect::<Option<_>>()?;
    let n = device.len().max(1) as u32;
    let mut mean = DeviceStageTimings::default();
    for stage in DeviceStage::ALL {
        mean.set(
            stage,
            device.iter().map(|t| t.get(stage)).sum::<Duration>() / n,
        );
    }
    Some(mean)
}

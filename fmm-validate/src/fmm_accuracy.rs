//! The accuracy of the complete FMM on a uniform tree (C3.2): the core of the
//! `fmm_accuracy` example.
//!
//! The problem ([`Problem`]): N points uniform in the cube [−1, 1)³, sources equal to
//! targets, and several charge vectors uniform in [−1, 1). The tree is uniform:
//! `max_level` = [`Config::max_level`] and one point per leaf as the refinement target,
//! so with enough points every leaf lies on the maximum level. [`run`] builds an
//! `nd_fmm_exec::fmm::Fmm` with gradients at degree p, evaluates every charge vector and
//! measures the output at a fixed sample of the targets against the [`Oracle`].
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
//! translation error (design §7, as re-derived for p = 18 in T9). f32 runs use charges rounded to f32 and an
//! oracle on the rounded charges, so the input rounding is not counted.
//!
//! [`metrics::ErrorNorms`]: crate::metrics::ErrorNorms

use std::time::Duration;

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{BuildTimings, FmmBuilder, ListSizes, StageTimings};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;

use crate::metrics::{ErrorAccumulator, ErrorNorms};
use crate::{SplitMix64, points};

/// The single-translation prediction of the relative L2 error of φ, P2M → M2L → L2P
/// pooled over the 316 V-list offsets (design §7, "Single-translation accuracy";
/// docs/phase3/README.md, "Predictions").
///
/// p = 3 and 8 are the Phase 1 T7 values (one draw of 1,000 sources, 1,000 targets per
/// offset). p = 18 is the T9 re-derivation, the median over nine draws with 10⁴
/// targets per offset; Phase 1's 2.71e-9 had too few targets for the heavy-tailed
/// error of the face offsets.
pub const PREDICTION: [(usize, f64); 3] = [(3, 1.77e-3), (8, 1.08e-5), (18, 5.78e-9)];

/// Returns the prediction at degree `p`, if [`PREDICTION`] has one.
pub fn prediction(p: usize) -> Option<f64> {
    PREDICTION.iter().find(|&&(q, _)| q == p).map(|&(_, e)| e)
}

/// The size of the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// The number of points, sources and targets alike.
    pub n: usize,
    /// The level of every leaf (`FmmBuilder::max_level`).
    pub max_level: usize,
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
        n: 100_000,
        max_level: 4,
        sampled: 1000,
        charge_vectors: 8,
        seed: 0xc32,
    };
}

/// The points, the sampled targets and the charge vectors of a [`Config`].
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The points, uniform in [−1, 1)³.
    pub points: Vec<[f64; 3]>,
    /// The positions in `points` of the sampled targets, distinct.
    pub sample: Vec<usize>,
    /// The charge vectors, uniform in [−1, 1).
    pub charges: Vec<Vec<f64>>,
}

impl Problem {
    /// Draws the problem: the points from `seed`, then the sample from the same
    /// generator (a partial Fisher–Yates shuffle), and charge vector k from
    /// `seed + 1 + k`.
    ///
    /// # Panics
    ///
    /// If `config.sampled > config.n`.
    pub fn new(config: &Config) -> Self {
        assert!(config.sampled <= config.n, "more samples than points");
        let mut rng = SplitMix64::new(config.seed);
        let points = points::cube(&mut rng, config.n, [0.0; 3], 1.0);
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
    /// Points per leaf: the smallest, the mean and the largest count.
    pub points_per_leaf: (usize, f64, usize),
    /// The sizes of the interaction lists.
    pub lists: ListSizes,
    /// The wall time of the build.
    pub build: BuildTimings,
    /// The wall time of each stage, the mean over the charge vectors.
    pub stages: StageTimings,
}

impl Run {
    /// The φ L2 error over the prediction at this p, if there is one.
    pub fn ratio(&self) -> Option<f64> {
        prediction(self.p).map(|e| self.potential.l2 / e)
    }
}

/// Builds the FMM of `problem` in precision `T` at degree `p` (default strategy,
/// gradients on), evaluates the charge vectors `charges` and measures the output
/// against `oracle` (module documentation).
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
/// If the FMM does not build or evaluate, for example on several ranks.
pub fn run<T: Stored + Equivalence + Default>(
    config: &Config,
    problem: &Problem,
    charges: &[Vec<T>],
    oracle: &Oracle,
    p: usize,
    comm: &SimpleCommunicator,
) -> Run {
    let builder = FmmBuilder::<T>::new(p)
        .max_level(config.max_level)
        .max_points_per_leaf(1)
        .gradients(true);
    let mut fmm = builder
        .build(&problem.points, &problem.points, comm)
        .unwrap_or_else(|error| panic!("the FMM does not build: {error}"));
    let counts = fmm.source_counts();
    let points_per_leaf = (
        counts.iter().copied().min().unwrap_or(0),
        counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64,
        counts.iter().copied().max().unwrap_or(0),
    );
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
        points_per_leaf,
        lists: fmm.list_sizes(),
        build: fmm.build_timings(),
        stages: mean(&stages),
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
    }
}

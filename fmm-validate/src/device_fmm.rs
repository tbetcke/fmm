//! The device FMM against the host path, stage by stage (Phase 4 T13; feature `gpu`): the
//! core of the `device_fmm` example.
//!
//! One [`measure`] builds the FMM of a [`Reference`] problem with [`Settings`] (backend,
//! threads, M2L strategy, tuning and table caches), measures its errors against the
//! direct sum over every charge vector as [`fmm_accuracy::measure`] does, and times it:
//!
//! - **evaluation**: the median wall time of [`Fmm::evaluate`] over `repeats` evaluations
//!   of the first charge vector, after the error evaluations (which compile the device
//!   kernels and warm the device);
//! - **stages**: on the host, the median of each stage of [`StageTimings`] over the same
//!   evaluations. With a device backend the default evaluation times only the enqueueing
//!   of each stage (docs/design/device-path.md §8.3), so a second FMM of the same settings
//!   is built with [`FmmBuilder::synchronous_stages`], which waits for the device after
//!   every stage, and the median of each stage over `repeats` of its evaluations is
//!   reported instead (one warm-up first). With a tuning cache that second build takes
//!   every choice from the cache the first one wrote, so both run the same choices;
//! - **build**: [`BuildTimings`] of the first build, the tables (loaded or built, by
//!   [`Fmm::cache_outcomes`]) and, with a tuning cache, the tuning time
//!   ([`TuningReport::time`], part of [`BuildTimings::device`]);
//! - **transfers, launches and syncs** of the last default evaluation ([`DeviceRun`]).
//!
//! The device's output is compared with a host run's over every target and charge
//! vector by [`compare`]: the relative L2 difference of φ and of ∇φ, the largest over the
//! vectors, the measure of the C4.8 gate (docs/phase4/README.md, "Accuracy measures":
//! within 1e-12 in f64 and 1e-5 in f32). [`Measurement::identical`] says whether two
//! evaluations of the first charge vector gave the same bits.
//!
//! The errors are those of [`fmm_accuracy`]: per charge vector the relative L2 and max
//! errors of φ and ∇φ at the sampled targets, then the root mean square over the vectors.
//! They depend on the settings only through what runs where (the device's kernels sum in
//! their own order), not on the threads or the timing mode. Timings are wall times on this
//! machine, for reports only: nothing here asserts one.
//!
//! [`TuningReport::time`]: nd_fmm_exec::tune::TuningReport::time
//! [`BuildTimings::device`]: nd_fmm_exec::fmm::BuildTimings::device

use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{Backend, BuildTimings, Fmm, FmmBuilder, FmmError, Output, StageTimings};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::{CacheOutcome, Stored};

use crate::calibration::{Precision, Reference};
use crate::fmm_accuracy::{self, Config, DeviceRun, Execution, Run};

/// Where and how a [`measure`] runs the FMM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// Where the operators run: the host path, or a device backend with every kind on the
    /// device.
    pub backend: Backend,
    /// Rayon threads of the host path (`FmmBuilder::threads`); with a GPU backend they
    /// serve host-fallback kinds only, and with `Backend::Cpu` they cap the units per cube.
    pub threads: usize,
    /// The M2L strategy (`FmmBuilder::strategy`). `Auto` resolves by the host rule on the
    /// host; on a device by the tuned choice with a [`tuning_cache`](Self::tuning_cache),
    /// else by the device's static rule.
    pub strategy: M2lStrategy,
    /// The tuning-cache directory of a device run (`FmmBuilder::tuning_cache`); `None`
    /// for no tuning.
    pub tuning_cache: Option<PathBuf>,
    /// The table-cache directory (`FmmBuilder::table_cache`); `None` builds the tables.
    pub table_cache: Option<PathBuf>,
    /// The timed evaluations, at least 1.
    pub repeats: usize,
}

impl Settings {
    /// The host path with `threads` threads and `strategy`, the default P2P kernel
    /// (`P2pChoice::Auto`), no table cache and 5 timed evaluations.
    pub fn host(threads: usize, strategy: M2lStrategy) -> Self {
        Self {
            backend: Backend::Host,
            threads,
            strategy,
            tuning_cache: None,
            table_cache: None,
            repeats: 5,
        }
    }

    /// `backend` with every kind on the device, one thread, `strategy`, no caches and 5
    /// timed evaluations.
    pub fn device(backend: Backend, strategy: M2lStrategy) -> Self {
        Self {
            threads: 1,
            backend,
            ..Self::host(1, strategy)
        }
    }

    /// The settings of an FMM of `config`'s tree at degree `p`, with gradients.
    pub fn builder<T>(&self, config: &Config, p: usize) -> FmmBuilder<T> {
        let execution = Execution {
            backend: self.backend,
            ..Execution::threads(self.threads)
        };
        let mut builder = fmm_accuracy::builder::<T>(config, p, execution).strategy(self.strategy);
        if let Some(dir) = &self.table_cache {
            builder = builder.table_cache(dir);
        }
        if let Some(dir) = &self.tuning_cache {
            builder = builder.tuning_cache(dir);
        }
        builder
    }

    /// Whether the M2L strategy is tuned: `Auto` on a device with a tuning cache.
    pub fn tuned(&self) -> bool {
        self.backend.is_device()
            && self.strategy == M2lStrategy::Auto
            && self.tuning_cache.is_some()
    }
}

impl fmt::Display for Settings {
    /// "host, 12 threads, Auto", "metal, tuned" or "metal, Rotation".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.backend {
            Backend::Host => write!(
                f,
                "host, {} thread{}, {:?}",
                self.threads,
                if self.threads == 1 { "" } else { "s" },
                self.strategy
            ),
            backend if self.tuned() => write!(f, "{backend}, tuned"),
            backend if self.strategy == M2lStrategy::Auto => write!(f, "{backend}, static rule"),
            backend => write!(f, "{backend}, {:?}", self.strategy),
        }
    }
}

/// One output in f64, in the caller's target order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    /// φ at every target.
    pub potential: Vec<f64>,
    /// ∇φ at every target.
    pub gradient: Vec<[f64; 3]>,
}

impl Values {
    /// The values of `output`, which must have gradients.
    fn of<T: RealScalar>(output: &Output<T>) -> Self {
        Self {
            potential: output
                .potential
                .iter()
                .map(|&v| RealScalar::to_f64(v))
                .collect(),
            gradient: output
                .gradient
                .as_ref()
                .expect("built with gradients")
                .iter()
                .map(|g| g.map(RealScalar::to_f64))
                .collect(),
        }
    }
}

/// The tuning of a build with a tuning cache.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    /// The time spent tuning, part of the build's device time.
    pub time: Duration,
    /// Whether any candidate was timed (else every choice came from the cache or the
    /// static rule).
    pub tuned: bool,
    /// Every decision as "decision: choice (source)", in the order taken.
    pub choices: Vec<String>,
}

/// The result of one [`measure`].
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// The problem and its tree.
    pub config: Config,
    /// The settings.
    pub settings: Settings,
    /// The errors, the tree, the build timings and, on a device, what ran where and what
    /// the last evaluation moved ([`Run::device`]); its `stages` are those of the error
    /// evaluations, the first of which compiles the device kernels: use
    /// [`stages`](Self::stages).
    pub run: Run,
    /// The median wall time of one [`Fmm::evaluate`].
    pub evaluation: Duration,
    /// The median of each stage: on the host over the timed evaluations, on a device
    /// over those of the synchronous-stages build.
    pub stages: StageTimings,
    /// Whether [`stages`](Self::stages) come from a build with a sync after every stage
    /// (every device run).
    pub synchronous: bool,
    /// The build's tables: whether every family was loaded from the table cache.
    pub tables_loaded: bool,
    /// The tuning, with a tuning cache on a device.
    pub tuning: Option<Tuning>,
    /// On a device, the bytes of every buffer the device operator allocates
    /// (`DeviceReport::memory_needed`).
    pub memory: Option<u64>,
    /// The output of every charge vector, for [`compare`]; may be cleared after use.
    pub outputs: Vec<Values>,
    /// Whether two evaluations of the first charge vector gave the same bits.
    pub identical: bool,
}

impl Measurement {
    /// The build timings.
    pub fn build(&self) -> &BuildTimings {
        &self.run.build
    }

    /// The far field (upward and downward), the leaf stage and the sum of the stages, in
    /// seconds, from [`stages`](Self::stages).
    pub fn far_near(&self) -> (f64, f64, f64) {
        let s = &self.stages;
        (
            (s.upward_local + s.upward_global + s.downward).as_secs_f64(),
            s.evaluate_leaves.as_secs_f64(),
            s.total().as_secs_f64(),
        )
    }
}

/// Builds and measures the FMM of `reference` (with `config`'s tree) in `precision` at
/// degree `p` with `settings` (module documentation).
///
/// # Errors
///
/// The error of [`FmmBuilder::build`], for example a configuration that does not fit in
/// the device's memory, or f64 on a device without f64 arithmetic.
///
/// # Collective operation
///
/// On `comm`, which must have one rank.
///
/// # Panics
///
/// If an evaluation fails, or `settings.repeats` is 0.
pub fn measure(
    config: &Config,
    reference: &Reference,
    (precision, p): (Precision, usize),
    settings: &Settings,
    comm: &SimpleCommunicator,
) -> Result<Measurement, FmmError> {
    match precision {
        Precision::F64 => measure_in::<f64>(
            config,
            reference,
            &reference.problem.charges,
            p,
            settings,
            comm,
        ),
        Precision::F32 => {
            measure_in::<f32>(config, reference, &reference.charges32, p, settings, comm)
        }
    }
}

/// [`measure`] in `T`, with the charges of `T`.
fn measure_in<T: Stored + SimdScalar + Equivalence + Default + RealScalar>(
    config: &Config,
    reference: &Reference,
    charges: &[Vec<T>],
    p: usize,
    settings: &Settings,
    comm: &SimpleCommunicator,
) -> Result<Measurement, FmmError> {
    assert!(settings.repeats > 0, "at least one timed evaluation");
    let oracle = if size_of::<T>() == 4 {
        &reference.oracle32
    } else {
        &reference.oracle64
    };
    let problem = &reference.problem;
    let builder = settings.builder::<T>(config, p);
    let (mut run, mut fmm) =
        fmm_accuracy::measure(config, problem, charges, oracle, &builder, comm)?;
    let tables_loaded = !fmm.cache_outcomes().is_empty()
        && fmm
            .cache_outcomes()
            .iter()
            .all(|(_, outcome)| matches!(outcome, CacheOutcome::Loaded));
    let tuning = fmm
        .device_report()
        .and_then(|report| report.tuning.as_ref())
        .filter(|report| report.file.is_some())
        .map(|report| Tuning {
            time: report.time,
            tuned: report.tuned(),
            choices: report
                .decisions
                .iter()
                .map(|d| format!("{}: {} ({})", d.decision, d.choice, d.source))
                .collect(),
        });

    let memory = fmm.device_report().map(|report| report.memory_needed);
    let mut outputs = Vec::with_capacity(charges.len());
    for q in charges {
        outputs.push(Values::of(&evaluate(&mut fmm, q)));
    }
    let first = &charges[0];
    let mut times = Vec::with_capacity(settings.repeats);
    let mut stages = Vec::with_capacity(settings.repeats);
    let mut identical = true;
    for _ in 0..settings.repeats {
        let start = Instant::now();
        let output = evaluate(&mut fmm, first);
        times.push(start.elapsed());
        identical &= Values::of(&output) == outputs[0];
        stages.push(output.timings);
    }
    run.device = DeviceRun::of(&fmm);
    drop(fmm);

    let synchronous = settings.backend.is_device();
    if synchronous {
        let mut timed =
            builder
                .synchronous_stages(true)
                .build(&problem.points, &problem.points, comm)?;
        evaluate(&mut timed, first);
        stages = (0..settings.repeats)
            .map(|_| evaluate(&mut timed, first).timings)
            .collect();
    }
    Ok(Measurement {
        config: *config,
        settings: settings.clone(),
        run,
        evaluation: median(times),
        stages: median_stages(&stages),
        synchronous,
        tables_loaded,
        tuning,
        memory,
        outputs,
        identical,
    })
}

/// One evaluation of `charges`.
fn evaluate<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &mut Fmm<'_, T>,
    charges: &[T],
) -> Output<T> {
    fmm.evaluate(charges)
        .unwrap_or_else(|error| panic!("the FMM does not evaluate: {error}"))
}

/// The median of `times` (the upper one of an even count).
fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

/// The median of each stage over `timings`; no device stage times.
fn median_stages(timings: &[StageTimings]) -> StageTimings {
    let stage = |f: fn(&StageTimings) -> Duration| median(timings.iter().map(f).collect());
    StageTimings {
        load: stage(|t| t.load),
        exchange_sources: stage(|t| t.exchange_sources),
        upward_local: stage(|t| t.upward_local),
        upward_global: stage(|t| t.upward_global),
        exchange_multipoles: stage(|t| t.exchange_multipoles),
        downward: stage(|t| t.downward),
        evaluate_leaves: stage(|t| t.evaluate_leaves),
        output: stage(|t| t.output),
        device: None,
        kinds: None,
    }
}

/// The relative L2 difference of `got`'s outputs from `want`'s over every target, of φ
/// and of ∇φ (the Euclidean norm per target), the largest over the charge vectors: the
/// device-against-host measure of the C4.8 gate.
///
/// # Panics
///
/// If either has no outputs, or they differ in the number of vectors or targets.
pub fn compare(got: &Measurement, want: &Measurement) -> (f64, f64) {
    assert!(
        !got.outputs.is_empty() && got.outputs.len() == want.outputs.len(),
        "the outputs of the same charge vectors"
    );
    got.outputs
        .iter()
        .zip(&want.outputs)
        .map(|(g, w)| {
            assert_eq!(g.potential.len(), w.potential.len(), "the same targets");
            let relative = |pairs: &mut dyn Iterator<Item = (f64, f64)>| {
                let (difference, norm) = pairs.fold((0.0, 0.0), |(d, n), (a, b)| {
                    (d + (a - b).powi(2), n + b * b)
                });
                (difference / norm).sqrt()
            };
            let potential =
                relative(&mut g.potential.iter().copied().zip(w.potential.iter().copied()));
            let gradient = relative(
                &mut g
                    .gradient
                    .iter()
                    .zip(&w.gradient)
                    .flat_map(|(a, b)| (0..3).map(move |k| (a[k], b[k]))),
            );
            (potential, gradient)
        })
        .fold((0.0, 0.0), |(p, g), (a, b)| {
            (f64::max(p, a), f64::max(g, b))
        })
}

/// The ratio of each of the four errors of `got` (φ L2, φ max, ∇φ L2, ∇φ max) to the same
/// error of `want`.
pub fn error_ratios(got: &Run, want: &Run) -> [f64; 4] {
    [
        got.potential.l2 / want.potential.l2,
        got.potential.max / want.potential.max,
        got.gradient.l2 / want.gradient.l2,
        got.gradient.max / want.gradient.max,
    ]
}

/// The FMM bounds of a device output against the host output of the same settings
/// (docs/phase4/README.md, "Accuracy measures"): the relative L2 difference of φ and ∇φ
/// (1e-12 in f64, 1e-5 in f32), and how far each error may move relative to the host's
/// (0.1% in f64, 5% in f32).
pub fn bounds(precision: Precision) -> (f64, f64) {
    match precision {
        Precision::F64 => (1e-12, 1e-3),
        Precision::F32 => (1e-5, 5e-2),
    }
}

//! The problem ([`Problem`]) and the measurement of every combination ([`run`]).
//!
//! Per combination (crate documentation, "Measurement"): one build and its
//! `BuildTimings`; `--warmup` untimed evaluations; `--repeats` timed evaluations with kind
//! timings off, whose outputs are compared bit for bit with the first; the errors of the
//! first against the f64 direct sum; on a device, what ran where and what one evaluation
//! moved. Unless `--kinds off`, the first `Fmm` is dropped and a second one is built with
//! the kind-timing mode, warmed up and timed the same way. The charges are the same in
//! every evaluation.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{
    Fmm, FmmBuilder, FmmError, KindTiming, KindTimings, OperatorKind, Output, StageTimings,
};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_tables::cache::{CacheOutcome, Stored};
use nd_fmm_validate::calibration::Precision;
use nd_fmm_validate::fmm_accuracy::{self, Oracle};
use nd_fmm_validate::metrics::ErrorAccumulator;
use nd_fmm_validate::{SplitMix64, points};

use crate::options::{Combination, Options};
use crate::report::{
    DeviceFacts, KIND_ORDER, KindOutcome, KindRow, Kinds, Measured, Outcome, Report, Row, Stats,
    Tree, markdown, title,
};

/// The seed of the points, the charges and the sampled targets.
pub const SEED: u64 = 0x00be_9c06;

/// The value types of the benchmark, f32 and f64: the bounds of `Fmm<T>`.
trait Value: Stored + SimdScalar + Equivalence + Default {}

impl<T: Stored + SimdScalar + Equivalence + Default> Value for T {}

/// The problem of one N: N points uniform in [0, 1)³ (`points::cube` about (½, ½, ½) with
/// half-width ½), then N charges uniform in [−1, 1) and the sampled targets (a partial
/// Fisher–Yates shuffle), all from one generator seeded with [`SEED`]. The points and
/// charges do not depend on the number of sampled targets.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The points, the sampled targets and the one charge vector, in the form of
    /// `fmm_accuracy`, whose `Oracle` computes the direct sum.
    inner: fmm_accuracy::Problem,
}

impl Problem {
    /// Draws the problem of `n` points with `sampled` targets (at most `n`).
    pub fn new(n: usize, sampled: usize) -> Self {
        let mut rng = SplitMix64::new(SEED);
        let points = points::cube(&mut rng, n, [0.5; 3], 0.5);
        let charges = points::charges(&mut rng, n);
        let sampled = sampled.min(n);
        let mut order: Vec<usize> = (0..n).collect();
        for i in 0..sampled {
            // Uniform in i..n from the top 32 bits (n ≤ 2³², a negligible bias).
            let j = i + (((rng.next_u64() >> 32) * (n - i) as u64) >> 32) as usize;
            order.swap(i, j);
        }
        order.truncate(sampled);
        Self {
            inner: fmm_accuracy::Problem {
                points,
                sample: order,
                charges: vec![charges],
            },
        }
    }

    /// The points, sources and targets alike.
    pub fn points(&self) -> &[[f64; 3]] {
        &self.inner.points
    }

    /// The charges in f64.
    pub fn charges(&self) -> &[f64] {
        &self.inner.charges[0]
    }

    /// The positions of the sampled targets in [`points`](Self::points).
    pub fn sample(&self) -> &[usize] {
        &self.inner.sample
    }

    /// The exact φ and ∇φ at the sampled targets, of the charges as the FMM of `precision`
    /// sees them (rounded to f32 for f32): the f64 direct sum over every point, divided
    /// by 4π (`fmm_accuracy::Oracle`). O(N × sampled), on one thread.
    pub fn oracle(&self, precision: Precision) -> Oracle {
        match precision {
            Precision::F64 => Oracle::new(&self.inner, &self.inner.charges),
            Precision::F32 => {
                let rounded: Vec<Vec<f64>> = self
                    .inner
                    .charges_as::<f32>()
                    .iter()
                    .map(|q| q.iter().map(|&v| f64::from(v)).collect())
                    .collect();
                Oracle::new(&self.inner, &rounded)
            }
        }
    }

    /// The charges rounded to `T`.
    fn charges_as<T: Value>(&self) -> Vec<T> {
        self.inner
            .charges_as::<T>()
            .pop()
            .expect("one charge vector")
    }
}

/// The settings of a combination's FMM: degree p, gradients, the tree, threads, strategy,
/// backend and caches of `options`; kind timings off.
pub fn builder<T>(options: &Options, combination: &Combination) -> FmmBuilder<T> {
    let mut builder = FmmBuilder::<T>::new(combination.p)
        .gradients(options.gradients)
        .threads(options.threads())
        .strategy(options.strategy)
        .backend(combination.backend)
        .output_pass(options.output_pass);
    if let Some(level) = options.max_level {
        builder = builder.max_level(level);
    }
    if let Some(size) = options.leaf_size {
        builder = builder.max_points_per_leaf(size);
    }
    if let Some(dir) = &options.table_cache {
        builder = builder.table_cache(dir);
    }
    if let Some(dir) = &options.tuning_cache {
        builder = builder.tuning_cache(dir);
    }
    builder
}

/// Measures every combination of `report.options` into `report.rows`, rewriting the
/// Markdown file `output` after each (its directory is created), with progress on
/// stderr. The problem of each N and the oracle of each (N, precision) are computed once.
///
/// # Errors
///
/// An error writing `output`; the rows measured so far stay in `report`.
///
/// # Collective operation
///
/// On `comm`, which must have one rank; with more than one thread, MPI must provide
/// `Threading::Funneled`.
pub fn run(report: &mut Report, output: &Path, comm: &SimpleCommunicator) -> io::Result<()> {
    if let Some(dir) = output.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let options = report.options.clone();
    let sampled = options.accuracy.unwrap_or(0);
    let mut problems: HashMap<usize, Problem> = HashMap::new();
    let mut oracles: HashMap<(usize, &'static str), Oracle> = HashMap::new();
    let combinations = options.combinations();
    let total = combinations.len();
    for (i, combination) in combinations.into_iter().enumerate() {
        let start = Instant::now();
        let n = combination.n;
        let problem = problems.entry(n).or_insert_with(|| {
            let start = Instant::now();
            let problem = Problem::new(n, sampled);
            eprintln!("N = {n}: points in {:.1} s", start.elapsed().as_secs_f64());
            problem
        });
        let oracle = options.accuracy.map(|_| {
            &*oracles
                .entry((n, combination.precision.name()))
                .or_insert_with(|| {
                    let start = Instant::now();
                    let oracle = problem.oracle(combination.precision);
                    eprintln!(
                        "N = {n}, {}: direct sum at {} targets in {:.1} s",
                        combination.precision.name(),
                        problem.sample().len(),
                        start.elapsed().as_secs_f64()
                    );
                    oracle
                })
        });
        eprintln!("[{}/{total}] {} ...", i + 1, title(&combination));
        let outcome = match combination.precision {
            Precision::F32 => measure::<f32>(&options, &combination, problem, oracle, comm),
            Precision::F64 => measure::<f64>(&options, &combination, problem, oracle, comm),
        };
        eprintln!(
            "[{}/{total}] {}: {} ({:.1} s)",
            i + 1,
            title(&combination),
            match &outcome {
                Outcome::Refused(error) => format!("refused: {error}"),
                Outcome::Failed(error) => format!("failed: {error}"),
                Outcome::Measured(m) => format!("mean {:.3} ms", m.evaluation.mean * 1e3),
            },
            start.elapsed().as_secs_f64()
        );
        report.rows.push(Row {
            combination,
            outcome,
        });
        std::fs::write(output, markdown(report))?;
    }
    Ok(())
}

/// Steps 1–6 of one combination in `T`.
fn measure<T: Value>(
    options: &Options,
    combination: &Combination,
    problem: &Problem,
    oracle: Option<&Oracle>,
    comm: &SimpleCommunicator,
) -> Outcome {
    let charges: Vec<T> = problem.charges_as();
    let builder = builder::<T>(options, combination);
    // Step 1: the build.
    let mut fmm = match builder.build(problem.points(), problem.points(), comm) {
        Ok(fmm) => fmm,
        Err(error) => return Outcome::Refused(error.to_string()),
    };
    let build = fmm.build_timings();
    let tables_loaded = !fmm.cache_outcomes().is_empty()
        && fmm
            .cache_outcomes()
            .iter()
            .all(|(_, outcome)| matches!(outcome, CacheOutcome::Loaded));
    // Step 2: the warm-up, into the reused output with `--reuse-output`.
    let mut reused = empty_output();
    for _ in 0..options.warmup {
        if let Err(error) = evaluate(&mut fmm, &charges, options.reuse_output, &mut reused) {
            return Outcome::Failed(error.to_string());
        }
    }
    // Step 3: the timed evaluations, each compared with the first.
    let mut times = Vec::with_capacity(options.repeats);
    let mut first: Option<Output<T>> = None;
    let mut identical = true;
    for _ in 0..options.repeats {
        let start = Instant::now();
        let output = match evaluate(&mut fmm, &charges, options.reuse_output, &mut reused) {
            Ok(output) => output,
            Err(error) => return Outcome::Failed(error.to_string()),
        };
        times.push(start.elapsed().as_secs_f64());
        let output = output.unwrap_or_else(|| reused.clone());
        match &first {
            None => first = Some(output),
            Some(first) => identical &= same_bits(first, &output),
        }
    }
    let first = first.expect("at least one timed evaluation");
    // Step 5: the accuracy of the first timed evaluation.
    let errors = oracle.map(|oracle| errors(&first, problem.sample(), oracle));
    drop(first);
    // Step 6: what ran where, and what the last evaluation moved.
    let device = device_facts(&fmm);
    let counts = fmm.source_counts();
    let lists = fmm.list_sizes();
    let tree = Tree {
        leaves: fmm.nleaves(),
        levels: fmm.nlevels(),
        points_per_leaf: (
            counts.iter().copied().min().unwrap_or(0),
            counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64,
            counts.iter().copied().max().unwrap_or(0),
        ),
        lists: [lists.u, lists.v, lists.w, lists.x],
    };
    let strategy = device_strategy(&fmm).unwrap_or_else(|| format!("{:?}", fmm.strategy()));
    let p2p = fmm.p2p_kernel().to_string();
    let threading = fmm.threading().to_string();
    drop(fmm);
    // Step 4: the kinds, from a second build.
    let kinds = match options.kinds {
        KindTiming::Off => KindOutcome::Off,
        mode => kinds(
            &builder.kind_timings(mode),
            problem,
            &charges,
            options,
            comm,
        ),
    };
    Outcome::Measured(Box::new(Measured {
        strategy,
        p2p,
        threading,
        tree,
        build: (
            build.total().as_secs_f64(),
            build.tables.as_secs_f64(),
            build.device.as_secs_f64(),
        ),
        tables_loaded,
        evaluation: Stats::of(&times),
        identical,
        errors,
        kinds,
        device,
    }))
}

/// An output without values, for [`evaluate`] to fill.
fn empty_output<T>() -> Output<T> {
    Output {
        potential: Vec::new(),
        gradient: None,
        timings: StageTimings::default(),
    }
}

/// One evaluation of `charges`: with `reuse` (`--reuse-output`) into `reused` by
/// `Fmm::evaluate_into`, returning `None`; otherwise by `Fmm::evaluate`, returning the
/// fresh output, so that the timed region holds the evaluation alone, as before T9.
fn evaluate<T: Value>(
    fmm: &mut Fmm<'_, T>,
    charges: &[T],
    reuse: bool,
    reused: &mut Output<T>,
) -> Result<Option<Output<T>>, FmmError> {
    if reuse {
        fmm.evaluate_into(charges, reused).map(|()| None)
    } else {
        fmm.evaluate(charges).map(Some)
    }
}

/// Whether two outputs have the same bits, φ and ∇φ.
fn same_bits<T: Value>(a: &Output<T>, b: &Output<T>) -> bool {
    let bits = |v: &T| (*v).to_f64().to_bits();
    let values =
        |x: &[T], y: &[T]| x.len() == y.len() && x.iter().zip(y).all(|(p, q)| bits(p) == bits(q));
    values(&a.potential, &b.potential)
        && match (&a.gradient, &b.gradient) {
            (Some(x), Some(y)) => values(x.as_flattened(), y.as_flattened()),
            (None, None) => true,
            _ => false,
        }
}

/// The relative L2 errors of φ and ∇φ of `output` at the sampled targets `sample`
/// against `oracle` (its one charge vector), as `fmm_accuracy` measures them.
fn errors<T: Value>(output: &Output<T>, sample: &[usize], oracle: &Oracle) -> (f64, Option<f64>) {
    let (phi, grad) = (&oracle.potential[0], &oracle.gradient[0]);
    let mut potential = ErrorAccumulator::new();
    let mut gradient = ErrorAccumulator::new();
    for (r, &i) in sample.iter().enumerate() {
        potential.add_values(&output.potential[i..=i], &phi[r..=r]);
        if let Some(g) = &output.gradient {
            gradient.add_vectors(&g[i..=i], &grad[r..=r]);
        }
    }
    (
        potential.finish().l2,
        output.gradient.as_ref().map(|_| gradient.finish().l2),
    )
}

/// Step 4: builds the FMM of `builder` (with a kind-timing mode), runs the warm-up and
/// the timed evaluations, and gathers the kinds.
fn kinds<T: Value>(
    builder: &FmmBuilder<T>,
    problem: &Problem,
    charges: &[T],
    options: &Options,
    comm: &SimpleCommunicator,
) -> KindOutcome {
    let mut fmm = match builder.build(problem.points(), problem.points(), comm) {
        Ok(fmm) => fmm,
        Err(error) => return KindOutcome::Refused(error.to_string()),
    };
    let mut reused = empty_output();
    for _ in 0..options.warmup {
        if let Err(error) = evaluate(&mut fmm, charges, options.reuse_output, &mut reused) {
            return KindOutcome::Refused(error.to_string());
        }
    }
    let mut walls = Vec::with_capacity(options.repeats);
    let mut timings = Vec::with_capacity(options.repeats);
    for _ in 0..options.repeats {
        let start = Instant::now();
        match evaluate(&mut fmm, charges, options.reuse_output, &mut reused) {
            Ok(output) => {
                walls.push(start.elapsed().as_secs_f64());
                timings.push(output.map_or(reused.timings, |o| o.timings));
            }
            Err(error) => return KindOutcome::Refused(error.to_string()),
        }
    }
    let timed: Vec<_> = timings
        .iter()
        .filter_map(|t| Some((t.kinds?, t.remainder()?, t)))
        .collect();
    let Some((last, _, _)) = timed.last() else {
        return KindOutcome::Refused(
            "no evaluation had kind times (a timing window was not timed)".to_owned(),
        );
    };
    let mode = last.mode;
    let rows = KIND_ORDER
        .iter()
        .map(|&name| {
            let kind = OperatorKind::ALL
                .into_iter()
                .find(|k| k.name() == name)
                .expect("every kind of the report order is a kind");
            let times: Vec<f64> = timed
                .iter()
                .map(|(k, _, _)| k.get(kind).total().as_secs_f64())
                .collect();
            let stats = Stats::of(&times);
            KindRow {
                kind: kind.name(),
                placement: last.get(kind).placement.to_string(),
                calls: last.get(kind).calls,
                min: stats.min,
                mean: stats.mean,
                max: stats.max,
            }
        })
        .collect();
    let mean = |f: &dyn Fn(&(KindTimings, Duration, &StageTimings)) -> f64| {
        timed.iter().map(f).sum::<f64>() / timed.len() as f64
    };
    let stats = |f: fn(&StageTimings) -> Duration| {
        Stats::of(
            &timed
                .iter()
                .map(|(_, _, t)| f(t).as_secs_f64())
                .collect::<Vec<_>>(),
        )
    };
    KindOutcome::Measured(Kinds {
        mode,
        rows,
        load: stats(|t| t.load),
        output: stats(|t| t.output),
        download: mean(&|(_, _, t)| t.download.as_secs_f64()),
        other: mean(&|(_, remainder, t)| remainder.saturating_sub(t.load + t.output).as_secs_f64()),
        sum: mean(&|(k, _, _)| k.total().as_secs_f64()),
        evaluation: walls.iter().sum::<f64>() / walls.len() as f64,
        timed: timed.len(),
    })
}

/// On a device, the M2L strategy as the device runs it (`DeviceReport::strategy_name`).
#[cfg(feature = "gpu")]
fn device_strategy<T: Value>(fmm: &Fmm<'_, T>) -> Option<String> {
    fmm.device_report().map(|report| report.strategy_name())
}

/// Without the `gpu` feature every run is on the host.
#[cfg(not(feature = "gpu"))]
fn device_strategy<T: Value>(_fmm: &Fmm<'_, T>) -> Option<String> {
    None
}

/// On a device: the device, the placement, the layouts, every device level call of the
/// translations, the memory, the transfers of the last evaluation and the tuning.
#[cfg(feature = "gpu")]
fn device_facts<T: Value>(fmm: &Fmm<'_, T>) -> Option<DeviceFacts> {
    use nd_fmm_exec::fmm::Placement;

    let (report, counters) = (fmm.device_report()?, fmm.device_counters()?);
    let on_device: Vec<&str> = OperatorKind::ALL
        .into_iter()
        .filter(|&k| report.placement(k) == Placement::Device)
        .map(OperatorKind::name)
        .collect();
    let mut calls: Vec<String> = report
        .translations
        .iter()
        .map(|t| {
            format!(
                "{}{} level {}: {} pairs, {}, {} chunk{}{}",
                t.kind,
                t.pass.map_or(String::new(), |pass| format!(" ({pass:?})")),
                t.level,
                t.pairs,
                t.gemm,
                t.chunks,
                if t.chunks == 1 { "" } else { "s" },
                t.library_rejection
                    .as_ref()
                    .map_or(String::new(), |why| format!(" (library rejected: {why})"))
            )
        })
        .collect();
    calls.extend(report.rotations.iter().map(|r| {
        format!(
            "M2L (rotation) level {}: {} pairs, {} rows, {}",
            r.level, r.pairs, r.rows, report.rotation_layout
        )
    }));
    let c = counters.evaluation;
    let tuning = report
        .tuning
        .as_ref()
        .filter(|t| t.file.is_some())
        .map(|t| {
            format!(
                "{:.2} s{}; {}",
                t.time.as_secs_f64(),
                if t.tuned() { "" } else { ", nothing timed" },
                t.decisions
                    .iter()
                    .map(|d| format!("{}: {} ({})", d.decision, d.choice, d.source))
                    .collect::<Vec<_>>()
                    .join("; ")
            )
        });
    Some(DeviceFacts {
        device: report.info.to_string(),
        on_device: if on_device.is_empty() {
            "none".to_owned()
        } else {
            on_device.join(", ")
        },
        layouts: format!(
            "P2P {}; leaf {}; GEMM {}; rotation {}",
            report.p2p_layout, report.leaf_layout, report.gemm_layout, report.rotation_layout
        ),
        calls,
        memory: report.memory_needed,
        evaluation: format!(
            "uploads {} ({} B), downloads {} ({} B), launches {}, syncs {}, timing windows {}",
            c.uploads,
            c.upload_bytes,
            c.downloads,
            c.download_bytes,
            c.launches,
            c.syncs,
            c.windows
        ),
        tuning,
    })
}

/// Without the `gpu` feature every run is on the host.
#[cfg(not(feature = "gpu"))]
fn device_facts<T: Value>(_fmm: &Fmm<'_, T>) -> Option<DeviceFacts> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_problem_is_seeded_and_in_the_unit_cube() {
        // Error measure: exact equality; the problem is a function of the seed.
        let a = Problem::new(500, 20);
        assert_eq!(a, Problem::new(500, 20));
        assert_eq!(a.points().len(), 500);
        assert!(
            a.points()
                .iter()
                .flatten()
                .all(|&x| (0.0..1.0).contains(&x))
        );
        assert!(a.charges().iter().all(|&q| (-1.0..1.0).contains(&q)));
        // Distinct sampled targets; the points and charges do not depend on their count.
        let mut sample = a.sample().to_vec();
        sample.sort_unstable();
        sample.dedup();
        assert_eq!(sample.len(), 20);
        let b = Problem::new(500, 7);
        assert_eq!((a.points(), a.charges()), (b.points(), b.charges()));
        assert_eq!(Problem::new(5, 20).sample().len(), 5);
        assert!(Problem::new(5, 0).sample().is_empty());
    }
}

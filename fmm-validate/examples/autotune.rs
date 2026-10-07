//! The tuning tables of the device path (Phase 4 T12, C4.7; the `tune` module of
//! `nd-fmm-exec`): for the uniform cube and the Plummer sphere of the C3.3 workload
//! (N = 10⁵, `max_level` 16, 64 points per leaf, gradients on), per degree, the device
//! FMM built with a fresh tuning cache, every decision with its candidates' times, the
//! choice and where it came from; then a second build from the same cache (every tuned
//! decision taken from it, the same choices), and on a GPU the evaluation time of the
//! tuned FMM against the static rule's. Prints Markdown on stdout and progress on stderr.
//! Feature `gpu` (with a backend).
//!
//! Defaults: Metal in f32 at p = 3, 6 and 8; the CPU runtime in f64 at p = 4, 8, 12 and
//! 16. The CPU runtime is the correctness backend: its timings say nothing about a GPU
//! and exercise the machinery only. Every time is the median of five batches of at least
//! 10 ms of launches queued between syncs, after a warm-up launch (compilation excluded);
//! a decision's table lists its candidates in the order they were registered (the static
//! rule's first), with "not registered" for those the device or the input-precision guard
//! refuses and "skipped" for those past the budget. Nothing is asserted but that the
//! second build takes every tuned decision from the cache with the same choice (a
//! decision the first build left to the static rule past the deadline is tuned by the
//! second, and counted in the summary).
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example autotune -- \
//!     --device metal --tuning-cache DIR [--table-cache DIR]
//! cargo run --release -p nd-fmm-validate --features cpu --example autotune -- \
//!     --device cpu --tuning-cache DIR --table-cache DIR [--n 20000]
//! ```
//!
//! `--tuning-cache DIR` (required): each (problem, backend, precision, p) gets the
//! subdirectory `DIR/<problem>-<backend>-<precision>-p<pp>`, emptied first, so that every
//! run tunes. `--table-cache DIR` loads and stores the tables (at p ≥ 12 the dense
//! strategy candidate needs it). `--precision f32|f64`, `--degrees p,p,…`, `--budget s`
//! (default 10), `--n N` (default 10⁵), `--evaluate` / `--no-evaluate` (the evaluation
//! times; on by default on a GPU, off on the CPU runtime). Metal needs a process with GPU
//! access (outside the macOS sandbox). It initialises MPI and runs on one rank.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{Backend, FmmBuilder};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tune::{STATIC_F64_DENSE_MAX_P, Source, Timing, TuningReport, static_strategy};
use nd_fmm_kernels::Precision;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_validate::bench::{cores, cpu_model, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};

/// The command line.
struct Arguments {
    device: String,
    precision: Option<String>,
    degrees: Option<Vec<usize>>,
    tuning_cache: PathBuf,
    table_cache: Option<PathBuf>,
    budget: f64,
    n: usize,
    evaluate: Option<bool>,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: autotune --device cpu|metal|cuda --tuning-cache DIR [--table-cache DIR] \
             [--precision f32|f64] [--degrees p,p,...] [--budget s] [--n N] \
             [--evaluate|--no-evaluate]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        precision: None,
        degrees: None,
        tuning_cache: PathBuf::new(),
        table_cache: None,
        budget: 10.0,
        n: 100_000,
        evaluate: None,
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--device", [value, tail @ ..]) => {
                parsed.device = value.clone();
                tail
            }
            ("--precision", [value, tail @ ..]) => {
                parsed.precision = Some(value.clone());
                tail
            }
            ("--degrees", [value, tail @ ..]) => {
                parsed.degrees = Some(
                    value
                        .split(',')
                        .map(|p| p.parse().unwrap_or_else(|_| usage()))
                        .collect(),
                );
                tail
            }
            ("--tuning-cache", [value, tail @ ..]) => {
                parsed.tuning_cache = PathBuf::from(value);
                tail
            }
            ("--table-cache", [value, tail @ ..]) => {
                parsed.table_cache = Some(PathBuf::from(value));
                tail
            }
            ("--budget", [value, tail @ ..]) => {
                parsed.budget = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            ("--n", [value, tail @ ..]) => {
                parsed.n = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            ("--evaluate", _) => {
                parsed.evaluate = Some(true);
                tail
            }
            ("--no-evaluate", _) => {
                parsed.evaluate = Some(false);
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty() || parsed.tuning_cache.as_os_str().is_empty() {
        usage();
    }
    parsed
}

fn main() {
    let arguments = arguments();
    let backend: Backend = arguments.device.parse().unwrap_or_else(|error| {
        eprintln!("--device: {error}");
        std::process::exit(2);
    });
    let info = backend.probe().unwrap_or_else(|reason| {
        eprintln!("--device {backend}: {reason} (Metal needs a process with GPU access)");
        std::process::exit(1);
    });
    let gpu = backend != Backend::Cpu;
    let precision = match arguments.precision.as_deref() {
        Some("f32") => Precision::F32,
        Some("f64") => Precision::F64,
        Some(other) => {
            eprintln!("--precision {other}: expected f32 or f64");
            std::process::exit(2);
        }
        None if gpu => Precision::F32,
        None => Precision::F64,
    };
    let degrees = arguments.degrees.clone().unwrap_or(match precision {
        Precision::F32 => vec![3, 6, 8],
        Precision::F64 => vec![4, 8, 12, 16],
    });
    let evaluate = arguments.evaluate.unwrap_or(gpu);
    let universe = mpi::initialize().expect("MPI initialises once");
    let comm = universe.world();

    println!("# Device autotune tables (Phase 4 T12, C4.7)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {info} |");
    println!("| precision | {precision} |");
    println!(
        "| timing | each candidate: a warm-up launch and a sync (compilation excluded), then \
         the median of 5 batches of >= 10 ms of launches queued between syncs |"
    );
    println!(
        "| budget | {} s per build; a fresh tuning cache per (problem, p) |",
        arguments.budget
    );
    if !gpu {
        println!(
            "| note | the CubeCL CPU runtime is the correctness backend: these timings \
             exercise the machinery and say nothing about a GPU |"
        );
    }
    println!();
    println!(
        "Static rule (no tuning cache): {precision} `Dense` at every p{}.",
        match precision {
            Precision::F32 => String::new(),
            Precision::F64 => format!(
                " ≤ {STATIC_F64_DENSE_MAX_P}, `Rotation` from p = {} (provisional)",
                STATIC_F64_DENSE_MAX_P + 1
            ),
        }
    );
    println!();

    let mut summary = Vec::new();
    for (name, distribution) in [
        ("cube", Distribution::Cube),
        ("plummer", Distribution::Plummer),
    ] {
        let config = Config {
            n: arguments.n,
            sampled: 1,
            charge_vectors: 1,
            ..Config::c33(distribution)
        };
        eprintln!("{name}: drawing N = {}", config.n);
        let problem = Problem::new(&config);
        for &p in &degrees {
            let run = Run {
                backend,
                name,
                config: &config,
                problem: &problem,
                p,
                arguments: &arguments,
                evaluate,
                comm: &comm,
            };
            summary.push(match precision {
                Precision::F32 => run.tune::<f32>(),
                Precision::F64 => run.tune::<f64>(),
            });
        }
    }

    println!("## Summary");
    println!();
    println!(
        "| problem | p | strategy (static rule) | strategy (tuned) | tuned decisions | cut | \
         tuning s | build s | build from the cache s | evaluate static ms | evaluate tuned ms |"
    );
    println!("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for line in summary {
        println!("{line}");
    }
    println!();
    println!(
        "Backends run: {backend} ({precision}); not run: {}.",
        [
            (backend != Backend::Cpu).then_some("cpu"),
            (backend != Backend::Metal).then_some("metal"),
            (backend != Backend::Cuda).then_some("cuda"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ")
    );
}

/// One (problem, p).
struct Run<'a> {
    backend: Backend,
    name: &'static str,
    config: &'a Config,
    problem: &'a Problem,
    p: usize,
    arguments: &'a Arguments,
    evaluate: bool,
    comm: &'a SimpleCommunicator,
}

impl Run<'_> {
    /// Tunes, prints the tables, rebuilds from the cache, and returns the summary row.
    fn tune<T: Stored + SimdScalar + Equivalence + Default + RealScalar>(&self) -> String {
        let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
        let dir = self.arguments.tuning_cache.join(format!(
            "{}-{}-{precision}-p{:02}",
            self.name, self.backend, self.p
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut builder = FmmBuilder::<T>::new(self.p)
            .gradients(true)
            .max_level(self.config.max_level)
            .max_points_per_leaf(self.config.max_points_per_leaf)
            .backend(self.backend);
        if let Some(tables) = &self.arguments.table_cache {
            builder = builder.table_cache(tables);
        }
        let tuned_builder = builder
            .clone()
            .tuning_cache(&dir)
            .tuning_budget(Duration::from_secs_f64(self.arguments.budget));
        eprintln!("{}, p = {}: tuning", self.name, self.p);
        let points = &self.problem.points;
        let start = Instant::now();
        let mut tuned = tuned_builder
            .build(points, points, self.comm)
            .unwrap_or_else(|error| panic!("the tuned FMM does not build: {error}"));
        let build = start.elapsed();
        let report = tuned
            .device_report()
            .and_then(|r| r.tuning.clone())
            .expect("a device build reports its tuning");
        print_tables(self.name, self.p, &report, build, &dir);

        eprintln!("{}, p = {}: from the cache", self.name, self.p);
        let start = Instant::now();
        let cached = tuned_builder
            .build(points, points, self.comm)
            .expect("the FMM builds from the cache");
        let rebuild = start.elapsed();
        let again = cached
            .device_report()
            .and_then(|r| r.tuning.clone())
            .expect("a device build reports its tuning");
        // Every tuned decision from the cache with the same choice; a decision left to the
        // static rule past the deadline is tuned by this build.
        let mut late = 0;
        for d in &report.decisions {
            let e = again.decision(d.decision).expect("the same decisions");
            if d.source == Source::Tuned {
                assert_eq!(e.source, Source::Cached, "{}: from the cache", d.decision);
                assert_eq!(e.choice, d.choice, "{}: the cache's choice", d.decision);
            } else if e.source == Source::Tuned {
                late += 1;
            }
        }
        drop(cached);

        let (mut static_ms, mut tuned_ms) = ("–".to_owned(), "–".to_owned());
        if self.evaluate {
            let charges: Vec<T> = self.problem.charges[0]
                .iter()
                .map(|&q| T::from_f64(q))
                .collect();
            let mut fixed = builder
                .build(points, points, self.comm)
                .expect("the static-rule FMM builds");
            static_ms = format!("{:.2}", median_evaluation(&mut fixed, &charges) * 1e3);
            tuned_ms = format!("{:.2}", median_evaluation(&mut tuned, &charges) * 1e3);
        }
        let tuned_count = report
            .decisions
            .iter()
            .filter(|d| d.source == Source::Tuned)
            .count();
        let cut = report.decisions.iter().filter(|d| d.cut).count();
        let precision = if size_of::<T>() == 4 {
            Precision::F32
        } else {
            Precision::F64
        };
        format!(
            "| {} | {} | {:?} | {:?} | {tuned_count} | {cut} | {:.2} | {:.2} | {:.2}{} | \
             {static_ms} | {tuned_ms} |",
            self.name,
            self.p,
            static_strategy(precision, self.p),
            tuned.strategy(),
            report.time.as_secs_f64(),
            build.as_secs_f64(),
            rebuild.as_secs_f64(),
            if late > 0 {
                format!(" ({late} left past the deadline tuned then)")
            } else {
                String::new()
            }
        )
    }
}

/// The median wall time of five evaluations after one warm-up, in seconds.
fn median_evaluation<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &mut nd_fmm_exec::fmm::Fmm<'_, T>,
    charges: &[T],
) -> f64 {
    fmm.evaluate(charges).expect("it evaluates");
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = Instant::now();
            fmm.evaluate(charges).expect("it evaluates");
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[2]
}

/// The tables of one build's tuning.
fn print_tables(name: &str, p: usize, report: &TuningReport, build: Duration, dir: &Path) {
    println!("## {name}, p = {p}");
    println!();
    println!(
        "Tuning {:.2} s of a {:.0} s budget{}; build {:.2} s; cache `{}`: {}{}.",
        report.time.as_secs_f64(),
        report.budget.as_secs_f64(),
        if report.deadline_hit {
            ", deadline reached"
        } else {
            ""
        },
        build.as_secs_f64(),
        dir.display(),
        report.cache,
        match &report.stored {
            Some(Ok(())) => ", stored".to_owned(),
            Some(Err(error)) => format!(", not stored ({error})"),
            None => String::new(),
        }
    );
    println!();
    println!("| decision | level | pairs | candidate | time µs | |");
    println!("| --- | ---: | ---: | --- | ---: | --- |");
    for d in &report.decisions {
        let level = d.level.map_or("–".to_owned(), |l| l.to_string());
        if d.times.is_empty() {
            println!(
                "| {} | {level} | {} | {} | – | chosen ({}{}) |",
                d.decision,
                d.pairs,
                d.choice,
                d.source,
                d.note
                    .as_deref()
                    .map_or(String::new(), |n| format!("; {n}"))
            );
            continue;
        }
        let best = d.best();
        for (candidate, timing) in &d.times {
            let (time, mark) = match timing {
                Timing::Measured(t) => {
                    let ratio = best.map_or(1.0, |b| t.as_secs_f64() / b.as_secs_f64());
                    (
                        format!("{:.1}", t.as_secs_f64() * 1e6),
                        if *candidate == d.choice {
                            format!(
                                "**chosen** ({}{})",
                                d.source,
                                if d.cut { ", cut by the deadline" } else { "" }
                            )
                        } else {
                            format!("{ratio:.2}×")
                        },
                    )
                }
                other => ("–".to_owned(), other.to_string()),
            };
            println!(
                "| {} | {level} | {} | {candidate} | {time} | {mark} |",
                d.decision, d.pairs
            );
        }
    }
    println!();
}

//! The device FMM against the host path, stage by stage, and the device leaf-size study
//! (Phase 4 T13). Prints Markdown on stdout and progress on stderr. Feature `gpu` (with a
//! backend).
//!
//! The problems are those of T12 (`nd_fmm_validate::calibration::config`): the uniform
//! cube and the Plummer sphere, N = 10⁵ points, sources equal to targets, eight charge
//! vectors, an adaptive tree with `max_level` 16 and 64 points per leaf, gradients on,
//! against the f64 direct sum at 1,000 sampled targets; and the cube at N = 10⁶
//! (`--large`), if it fits in the device's memory.
//!
//! - **FMM** (`--part fmm`): per problem and degree (f32 at p = 3, 6 and 8 by default),
//!   the device with every kind on it, under the autotuned strategy (`Auto` with a fresh
//!   tuning cache, `--tuning-cache`) and under each fixed strategy (`Dense`, `Classes`,
//!   which the device runs as dense, and `Rotation`), against the host path
//!   (`P2pChoice::Auto`, the widest ISA) under each fixed strategy at one thread and at
//!   `--threads` (default: the performance cores). Per run the errors, the build (with
//!   the tables and the tuning apart), the stage times, the evaluation time and the
//!   speed-ups, and on the device the transfers, launches and syncs of an evaluation and
//!   the device's output against the host's (`nd_fmm_validate::device_fmm`).
//! - **Leaf size** (`--part leaf`): the device under its static rule (no tuning cache)
//!   with the refinement targets 16, 32, 64, 128 and 256 (`calibration::LEAF_SIZES`), the
//!   cube and the Plummer sphere at p = 3 and 8 (`--leaf-degrees`): the tree, the errors,
//!   the stage times and the near/far balance, the fastest size per distribution and p,
//!   and the device leaf-size rule (T13): the size with the smallest geometric mean of
//!   the evaluation time over the four configurations, kept against 64 only if at least
//!   5% faster and no φ or ∇φ L2 error there is more than 10% worse than at 64
//!   (`calibration::leaf_size_rule`).
//!
//! Timing: the evaluation time is the median wall time of `Fmm::evaluate` over
//! `--repeats` evaluations (default 5) of the first charge vector, after the eight error
//! evaluations, which compile the device kernels and warm the device; each device
//! evaluation queues every launch and syncs once, at its download. The stage times of a
//! device run come from a second build of the same settings with `synchronous_stages`
//! (a sync after each stage; the median per stage over `--repeats` evaluations), so they
//! add up to more than the evaluation; the host's are the medians of its own timed
//! evaluations. Upward = P2M and M2M, downward = L2L, M2L and P2L, leaves = L2P, M2P and
//! P2P; the design's timing (docs/design/device-path.md §8.3) resolves stages, not kinds:
//! M2L and P2P alone are in the kernel harnesses (`m2l_kernels`, `p2p_kernels`). Every
//! time is measured on this machine and reported, never asserted.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example device_fmm -- \
//!     --device metal --tuning-cache DIR --table-cache DIR
//! ```
//!
//! Options: `--device cpu|metal|cuda` (required); `--precision f32|f64` (default f32 on a
//! GPU, f64 on the CPU runtime); `--degrees p,p,…` (default 3,6,8 in f32, 8,12,18 in
//! f64); `--leaf-degrees p,p,…` (default 3,8); `--part fmm|leaf|all` (default all);
//! `--n N` (default 10⁵); `--large N` (the cube's extra size, default 10⁶, 0 for none);
//! `--threads n`; `--repeats r`; `--tuning-cache DIR` (without it the tuned rows are
//! left out; each (problem, precision, p) gets the subdirectory
//! `DIR/<problem>-<N>-<backend>-<precision>-p<pp>`, emptied first, so that every run
//! tunes); `--table-cache DIR` (the tables are loaded from it, or built and stored). The
//! CPU runtime is the correctness backend: its timings say nothing about a GPU. Metal
//! needs a process with GPU access (outside the macOS sandbox). It initialises MPI at
//! `Threading::Funneled` and runs on one rank.
//!
//! The CUDA run: fmm-validate/results/phase4-m3max.md, "The CUDA run". On locust's H100
//! (Phase 4S, by hand: `tools/gh200/remote.sh`, feature `cuda`), `--device cuda` runs in
//! f32 and f64; on Linux `--threads` defaults to `available_parallelism` (72 on Grace).

use std::path::PathBuf;
use std::time::Instant;

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::Backend;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::calibration::{
    LEAF_RULE_BASELINE, LEAF_RULE_ERROR_GROWTH, LEAF_RULE_GAIN, LEAF_SIZES, Precision, Reference,
    config, leaf_size_rule,
};
use nd_fmm_validate::device_fmm::{Measurement, Settings, bounds, compare, error_ratios, measure};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution};

/// The fixed strategies of the FMM part, on the host and on the device.
const FIXED: [M2lStrategy; 3] = [
    M2lStrategy::Dense,
    M2lStrategy::Classes,
    M2lStrategy::Rotation,
];

/// Which parts run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Fmm,
    Leaf,
    All,
}

/// The command line.
struct Arguments {
    backend: Backend,
    precision: Precision,
    degrees: Vec<usize>,
    leaf_degrees: Vec<usize>,
    part: Part,
    n: usize,
    large: usize,
    threads: usize,
    repeats: usize,
    tuning_cache: Option<PathBuf>,
    table_cache: Option<PathBuf>,
}

/// Parses the command line (module documentation); exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: device_fmm --device cpu|metal|cuda [--precision f32|f64] [--degrees p,p,...] \
             [--leaf-degrees p,p,...] [--part fmm|leaf|all] [--n N] [--large N] [--threads n] \
             [--repeats r] [--tuning-cache DIR] [--table-cache DIR]; got {args:?}"
        );
        std::process::exit(2);
    };
    let list = |value: &str| -> Vec<usize> {
        value
            .split(',')
            .map(|p| p.parse().unwrap_or_else(|_| usage()))
            .collect()
    };
    let number = |value: &str| -> usize { value.parse().unwrap_or_else(|_| usage()) };
    let (mut backend, mut precision, mut degrees, mut leaf_degrees) = (None, None, None, None);
    let mut parsed = Arguments {
        backend: Backend::Host,
        precision: Precision::F32,
        degrees: Vec::new(),
        leaf_degrees: Vec::new(),
        part: Part::All,
        n: 100_000,
        large: 1_000_000,
        threads: performance_cores().unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
        }),
        repeats: 5,
        tuning_cache: None,
        table_cache: None,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match (flag.as_str(), value.as_str()) {
            ("--device", b) => match b.parse::<Backend>() {
                Ok(b) if b.is_device() => backend = Some(b),
                _ => usage(),
            },
            ("--precision", "f32") => precision = Some(Precision::F32),
            ("--precision", "f64") => precision = Some(Precision::F64),
            ("--degrees", v) => degrees = Some(list(v)),
            ("--leaf-degrees", v) => leaf_degrees = Some(list(v)),
            ("--part", "fmm") => parsed.part = Part::Fmm,
            ("--part", "leaf") => parsed.part = Part::Leaf,
            ("--part", "all") => parsed.part = Part::All,
            ("--n", v) => parsed.n = number(v),
            ("--large", v) => parsed.large = number(v),
            ("--threads", v) => parsed.threads = number(v).max(1),
            ("--repeats", v) => parsed.repeats = number(v).max(1),
            ("--tuning-cache", v) => parsed.tuning_cache = Some(PathBuf::from(v)),
            ("--table-cache", v) => parsed.table_cache = Some(PathBuf::from(v)),
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() || parsed.n < 1000 {
        usage();
    }
    parsed.backend = backend.unwrap_or_else(|| usage());
    let gpu = parsed.backend != Backend::Cpu;
    parsed.precision = precision.unwrap_or(if gpu { Precision::F32 } else { Precision::F64 });
    parsed.degrees = degrees.unwrap_or(match parsed.precision {
        Precision::F32 => vec![3, 6, 8],
        Precision::F64 => vec![8, 12, 18],
    });
    parsed.leaf_degrees = leaf_degrees.unwrap_or(vec![3, 8]);
    parsed
}

fn main() {
    let arguments = arguments();
    let info = arguments.backend.probe().unwrap_or_else(|reason| {
        eprintln!(
            "--device {}: {reason} (Metal needs a process with GPU access)",
            arguments.backend
        );
        std::process::exit(1);
    });
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if comm.size() != 1 {
        eprintln!(
            "device_fmm runs on one rank (a one-rank benchmark; ranks would share the device)"
        );
        std::process::exit(2);
    }
    if arguments.threads > 1 && provided < Threading::Funneled {
        eprintln!(
            "--threads {} needs MPI at Funneled; it provides {provided:?}",
            arguments.threads
        );
        std::process::exit(2);
    }

    let start = Instant::now();
    let mut fmm_tables = Vec::new();
    if arguments.part != Part::Leaf {
        let mut problems = vec![
            config(Distribution::Cube, arguments.n),
            config(Distribution::Plummer, arguments.n),
        ];
        if arguments.large > 0 {
            problems.push(config(Distribution::Cube, arguments.large));
        }
        for problem in &problems {
            fmm_tables.push(fmm_part(problem, &arguments, &comm));
        }
    }
    let leaf_runs = if arguments.part == Part::Fmm {
        Vec::new()
    } else {
        leaf_part(&arguments, &comm)
    };
    let wall = start.elapsed();

    print_header(&arguments, &info.to_string(), wall.as_secs_f64());
    for tables in &fmm_tables {
        print_fmm(tables, &arguments);
    }
    if !fmm_tables.is_empty() {
        print_summary(&fmm_tables, &arguments);
    }
    if !leaf_runs.is_empty() {
        print_leaf_study(&leaf_runs, &arguments);
        print_leaf_rule(&leaf_runs, &arguments);
    }
    println!();
    println!(
        "Backends run: {} ({}); host ({}); not run: {}.",
        arguments.backend,
        arguments.precision.name(),
        cpu_model(),
        [
            (arguments.backend != Backend::Cpu).then_some("cpu"),
            (arguments.backend != Backend::Metal).then_some("metal"),
            (arguments.backend != Backend::Cuda).then_some("cuda"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(", ")
    );
}

/// The runs of one problem at every degree.
struct FmmTable {
    config: Config,
    /// Per degree: the host runs (each fixed strategy at 1 and n threads), then the device
    /// runs (tuned, then each fixed strategy).
    degrees: Vec<(usize, Vec<Measurement>, Vec<DeviceRow>)>,
}

/// A device run, with its output against the host's of the same strategy at one thread
/// (`device_fmm::compare`), or why it did not build.
struct DeviceRow {
    settings: Settings,
    measured: Result<(Measurement, (f64, f64)), String>,
}

/// The FMM part for one problem.
fn fmm_part(config: &Config, arguments: &Arguments, comm: &SimpleCommunicator) -> FmmTable {
    let name = problem_name(config);
    let reference = Reference::new(config);
    eprintln!(
        "{name}: problem and oracles in {:.1} s",
        reference.time.as_secs_f64()
    );
    let precision = arguments.precision;
    let mut degrees = Vec::new();
    for &p in &arguments.degrees {
        let mut host = Vec::new();
        for strategy in FIXED {
            for threads in [1, arguments.threads] {
                let settings = Settings {
                    table_cache: arguments.table_cache.clone(),
                    repeats: arguments.repeats,
                    ..Settings::host(threads, strategy)
                };
                let mut m = run(config, &reference, (precision, p), &settings, comm)
                    .expect("the host builds");
                // The device is compared with the host at one thread (the same bits).
                if threads != 1 {
                    m.outputs.clear();
                }
                host.push(m);
            }
        }
        let mut device = Vec::new();
        let tuned = arguments.tuning_cache.as_ref().map(|dir| {
            let dir = dir.join(format!(
                "{}-{}-{}-{}-p{p:02}",
                config.distribution.name(),
                config.n,
                arguments.backend,
                precision.name()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            (M2lStrategy::Auto, Some(dir))
        });
        for (strategy, tuning_cache) in tuned
            .into_iter()
            .chain(FIXED.into_iter().map(|s| (s, None)))
        {
            let settings = Settings {
                tuning_cache,
                table_cache: arguments.table_cache.clone(),
                repeats: arguments.repeats,
                ..Settings::device(arguments.backend, strategy)
            };
            let measured = run(config, &reference, (precision, p), &settings, comm)
                .map(|mut m| {
                    let comparison = compare(&m, host_of(&host, m.run.strategy, 1));
                    m.outputs.clear();
                    (m, comparison)
                })
                .map_err(|error| error.to_string());
            device.push(DeviceRow { settings, measured });
        }
        for m in &mut host {
            m.outputs.clear();
        }
        degrees.push((p, host, device));
    }
    FmmTable {
        config: *config,
        degrees,
    }
}

/// One measurement, with progress on stderr; the device's output is compared with the
/// host's later, so the outputs are kept.
fn run(
    config: &Config,
    reference: &Reference,
    (precision, p): (Precision, usize),
    settings: &Settings,
    comm: &SimpleCommunicator,
) -> Result<Measurement, nd_fmm_exec::fmm::FmmError> {
    let start = Instant::now();
    let result = measure(config, reference, (precision, p), settings, comm);
    match &result {
        Ok(m) => eprintln!(
            "{} {} p = {p}, {} per leaf, {settings}: φ L2 {:.3e}, evaluate {} ms ({:.1} s)",
            problem_name(config),
            precision.name(),
            config.max_points_per_leaf,
            m.run.potential.l2,
            ms(m.evaluation.as_secs_f64()),
            start.elapsed().as_secs_f64()
        ),
        Err(error) => eprintln!(
            "{} {} p = {p}, {settings}: does not build: {error}",
            problem_name(config),
            precision.name()
        ),
    }
    result
}

/// "cube, N = 100000".
fn problem_name(config: &Config) -> String {
    format!("{}, N = {}", config.distribution.name(), config.n)
}

/// Milliseconds with two decimals.
fn ms(seconds: f64) -> String {
    format!("{:.2}", seconds * 1e3)
}

/// The machine, the device and the conventions of the report.
fn print_header(arguments: &Arguments, device: &str, wall: f64) {
    println!("# The device FMM against the host path (Phase 4 T13)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!(
        "| machine | {}; {}{}; {}; {} |",
        cpu_model(),
        cores(),
        performance_cores().map_or(String::new(), |p| format!(", {p} performance")),
        target(),
        toolchain()
    );
    println!("| device | {device} |");
    println!(
        "| BLAS threads | {} |",
        [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"
        ]
        .iter()
        .map(|v| format!(
            "{v}={}",
            std::env::var(v).unwrap_or_else(|_| "unset".into())
        ))
        .collect::<Vec<_>>()
        .join(", ")
    );
    println!("| precision | {} |", arguments.precision.name());
    println!(
        "| host | `P2pChoice::Auto` at 1 and {} threads; one rank |",
        arguments.threads
    );
    println!(
        "| timing | evaluate: median wall time of {} evaluations of the first charge vector \
         after the eight error evaluations (compilation excluded; one sync per device \
         evaluation); stages: on the device the medians of a synchronous-stages build (a \
         sync after each stage), on the host the medians of the timed evaluations |",
        arguments.repeats
    );
    println!(
        "| tuning | {} |",
        arguments.tuning_cache.as_ref().map_or_else(
            || "no tuning cache: no tuned rows".to_owned(),
            |dir| format!(
                "a fresh tuning cache per (problem, p) under `{}`, the default budget",
                dir.display()
            )
        )
    );
    println!(
        "| tables | {} |",
        arguments.table_cache.as_ref().map_or_else(
            || "built at every build".to_owned(),
            |dir| format!("through the table cache `{}`", dir.display())
        )
    );
    println!("| wall time of every run | {wall:.0} s |");
    if arguments.backend == Backend::Cpu {
        println!(
            "| note | the CubeCL CPU runtime is the correctness backend: its timings say \
             nothing about a GPU |"
        );
    }
}

/// The host run of `strategy` at `threads`.
fn host_of(host: &[Measurement], strategy: M2lStrategy, threads: usize) -> &Measurement {
    host.iter()
        .find(|m| m.settings.strategy == strategy && m.settings.threads == threads)
        .expect("every fixed strategy runs on the host at both thread counts")
}

/// The host path's default: `Auto`, as the host resolves it at this p.
fn host_default(host: &[Measurement], threads: usize) -> &Measurement {
    let p = host[0].run.p;
    host_of(host, M2lStrategy::Auto.resolve(p), threads)
}

/// The tables of one problem.
fn print_fmm(table: &FmmTable, arguments: &Arguments) {
    let n = arguments.threads;
    println!();
    println!(
        "## {}, {} ({}), {} per leaf",
        problem_name(&table.config),
        table.config.distribution.description(),
        arguments.precision.name(),
        table.config.max_points_per_leaf
    );
    for (p, host, device) in &table.degrees {
        let first = &host[0].run;
        println!();
        println!(
            "### p = {p}: {} leaves on {} levels, points per leaf {} / {:.1} / {} (min / mean / \
             max), lists U {} V {} W {} X {}",
            first.nleaves,
            first.nlevels,
            first.points_per_leaf.0,
            first.points_per_leaf.1,
            first.points_per_leaf.2,
            first.lists.u,
            first.lists.v,
            first.lists.w,
            first.lists.x
        );
        println!();
        println!(
            "Times in ms. Build: total, of it the tables (L loaded from the cache, B built) and \
             the tuning. x 1 / x {n}: the evaluation's speed-up over the host with the same \
             strategy at 1 / {n} threads (the tuned row: over the host default, `Auto`)."
        );
        println!();
        println!(
            "| run | strategy | build | tables | tuning | upward | downward | leaves | near share \
             | evaluate | x 1 | x {n} |"
        );
        println!(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for m in host {
            print_timing_row(m, None);
        }
        for DeviceRow { settings, measured } in device {
            match measured {
                Ok((m, _)) => {
                    let base = if settings.tuned() {
                        (host_default(host, 1), host_default(host, n))
                    } else {
                        (
                            host_of(host, settings.strategy, 1),
                            host_of(host, settings.strategy, n),
                        )
                    };
                    print_timing_row(m, Some(base));
                }
                Err(error) => {
                    println!("| {settings} | – | does not build: {error} | | | | | | | | | |")
                }
            }
        }
        println!();
        println!(
            "Errors: root mean squares over the charge vectors of the relative L2 and max \
             errors at the sampled targets. Device rows: the four errors over the host's of \
             the same strategy at one thread (bound 1 ± {}), the relative L2 difference of the \
             output from that host's over every target and vector (bound {:.0e}), and an \
             evaluation's transfers, launches and syncs (every kind on the device).",
            bounds(arguments.precision).1,
            bounds(arguments.precision).0
        );
        println!();
        println!(
            "| run | φ L2 | φ max | ∇φ L2 | ∇φ max | errors / host | device − host φ, ∇φ | \
             up / down (bytes) | launches | syncs | bit-identical | device memory |"
        );
        println!(
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: |"
        );
        for m in host {
            if m.settings.threads != 1 {
                continue;
            }
            let r = &m.run;
            println!(
                "| {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | | | | | | {} | |",
                m.settings,
                r.potential.l2,
                r.potential.max,
                r.gradient.l2,
                r.gradient.max,
                yes(m.identical)
            );
        }
        for row in device {
            let Ok((m, (dp, dg))) = &row.measured else {
                continue;
            };
            let r = &m.run;
            let base = host_of(host, r.strategy, 1);
            let ratios = error_ratios(r, &base.run);
            let d = r.device.as_ref().expect("a device run");
            let [up, up_bytes, down, down_bytes, launches, syncs, _] = d.evaluation;
            println!(
                "| {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} | {dp:.2e}, {dg:.2e} | {up} × {up_bytes} / \
                 {down} × {down_bytes} | {launches} | {syncs} | {} | {:.1} MB |",
                m.settings,
                r.potential.l2,
                r.potential.max,
                r.gradient.l2,
                r.gradient.max,
                ratios
                    .iter()
                    .map(|x| format!("{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                yes(m.identical),
                m.memory.unwrap_or(0) as f64 / 1e6
            );
        }
        for DeviceRow { settings, measured } in device {
            let Ok((m, _)) = measured else { continue };
            let d = m.run.device.as_ref().expect("a device run");
            println!();
            println!(
                "- {settings}: {} ({}){}",
                d.strategy,
                d.gemms,
                m.tuning.as_ref().map_or(String::new(), |t| format!(
                    "; tuning {:.2} s{}: {}",
                    t.time.as_secs_f64(),
                    if t.tuned { "" } else { " (nothing timed)" },
                    t.choices.join("; ")
                )),
            );
        }
    }
}

/// "yes" or "**no**".
fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "**no**" }
}

/// One row of the timing table; `base` the host runs at 1 and n threads to compare with.
fn print_timing_row(m: &Measurement, base: Option<(&Measurement, &Measurement)>) {
    let s = &m.stages;
    let b = m.build();
    let (_, leaves, total) = m.far_near();
    let speedup = |h: &Measurement| {
        format!(
            "{:.1}",
            h.evaluation.as_secs_f64() / m.evaluation.as_secs_f64()
        )
    };
    println!(
        "| {} | {:?} | {} | {} {} | {} | {} | {} | {} | {:.0}% | {} | {} | {} |",
        m.settings,
        m.run.strategy,
        ms(b.total().as_secs_f64()),
        ms(b.tables.as_secs_f64()),
        if m.tables_loaded { "L" } else { "B" },
        m.tuning
            .as_ref()
            .map_or("–".to_owned(), |t| ms(t.time.as_secs_f64())),
        ms((s.upward_local + s.upward_global).as_secs_f64()),
        ms(s.downward.as_secs_f64()),
        ms(leaves),
        100.0 * leaves / total,
        ms(m.evaluation.as_secs_f64()),
        base.map_or("–".to_owned(), |(one, _)| speedup(one)),
        base.map_or("–".to_owned(), |(_, many)| speedup(many)),
    );
}

/// One line per problem and degree: the host default against the tuned device and the
/// fastest device row.
fn print_summary(tables: &[FmmTable], arguments: &Arguments) {
    let n = arguments.threads;
    println!();
    println!("## Summary: evaluation time");
    println!();
    println!(
        "Median ms. Host: its default (`Auto`, `P2pChoice::Auto`). Device: tuned, and the \
         fastest of its rows."
    );
    println!();
    println!(
        "| problem | p | host 1 thread | host {n} threads | device tuned | device fastest | \
         tuned x 1 | tuned x {n} |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |");
    for table in tables {
        for (p, host, device) in &table.degrees {
            let (one, many) = (host_default(host, 1), host_default(host, n));
            let tuned = device
                .iter()
                .find(|row| row.settings.tuned())
                .and_then(|row| row.measured.as_ref().ok())
                .map(|(m, _)| m);
            let fastest = device
                .iter()
                .filter_map(|row| row.measured.as_ref().ok())
                .map(|(m, _)| m)
                .min_by_key(|m| m.evaluation);
            let t = |m: &Measurement| m.evaluation.as_secs_f64();
            println!(
                "| {} | {p} | {} | {} | {} | {} | {} | {} |",
                problem_name(&table.config),
                ms(t(one)),
                ms(t(many)),
                tuned.map_or("–".to_owned(), |m| ms(t(m))),
                fastest.map_or("–".to_owned(), |m| format!(
                    "{} ({})",
                    ms(t(m)),
                    m.settings
                )),
                tuned.map_or("–".to_owned(), |m| format!("{:.1}", t(one) / t(m))),
                tuned.map_or("–".to_owned(), |m| format!("{:.1}", t(many) / t(m))),
            );
        }
    }
}

/// The leaf-size study: per distribution, degree and size, the device under its static
/// rule.
fn leaf_part(arguments: &Arguments, comm: &SimpleCommunicator) -> Vec<Measurement> {
    let mut runs = Vec::new();
    for distribution in [Distribution::Cube, Distribution::Plummer] {
        let base = config(distribution, arguments.n);
        let reference = Reference::new(&base);
        for &p in &arguments.leaf_degrees {
            for max_points_per_leaf in LEAF_SIZES {
                let config = Config {
                    max_points_per_leaf,
                    ..base
                };
                let settings = Settings {
                    table_cache: arguments.table_cache.clone(),
                    repeats: arguments.repeats,
                    ..Settings::device(arguments.backend, M2lStrategy::Auto)
                };
                let mut m = run(
                    &config,
                    &reference,
                    (arguments.precision, p),
                    &settings,
                    comm,
                )
                .unwrap_or_else(|error| panic!("the leaf-size run does not build: {error}"));
                m.outputs.clear();
                runs.push(m);
            }
        }
    }
    runs
}

/// The configurations of the leaf study, in a fixed order.
fn configurations(arguments: &Arguments) -> Vec<(Distribution, usize)> {
    [Distribution::Cube, Distribution::Plummer]
        .into_iter()
        .flat_map(|d| arguments.leaf_degrees.iter().map(move |&p| (d, p)))
        .collect()
}

/// The runs of one configuration, in the order of [`LEAF_SIZES`].
fn configuration(runs: &[Measurement], d: Distribution, p: usize) -> Vec<&Measurement> {
    LEAF_SIZES
        .iter()
        .map(|&size| {
            runs.iter()
                .find(|m| {
                    m.run.distribution == d && m.run.p == p && m.config.max_points_per_leaf == size
                })
                .expect("every size runs in every configuration")
        })
        .collect()
}

/// Every run of the study, then the fastest size per configuration.
fn print_leaf_study(runs: &[Measurement], arguments: &Arguments) {
    println!();
    println!(
        "## Device leaf-size study ({}, {}, static rule)",
        arguments.backend,
        arguments.precision.name()
    );
    println!();
    println!(
        "Times in ms: stages from the synchronous-stages build; far = upward + downward; \
         near share = leaves / sum of the stages; evaluate = the median of the default \
         build."
    );
    println!();
    println!(
        "| distribution | p | max points per leaf | leaves | points per leaf (min / mean / max) \
         | V pairs | φ L2 | ∇φ L2 | far | leaves | near share | evaluate |"
    );
    println!(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for m in runs {
        let r = &m.run;
        let (far, leaves, total) = m.far_near();
        println!(
            "| {} | {} | {} | {} | {} / {:.1} / {} | {} | {:.3e} | {:.3e} | {} | {} | {:.0}% | {} |",
            r.distribution.name(),
            r.p,
            m.config.max_points_per_leaf,
            r.nleaves,
            r.points_per_leaf.0,
            r.points_per_leaf.1,
            r.points_per_leaf.2,
            r.lists.v,
            r.potential.l2,
            r.gradient.l2,
            ms(far),
            ms(leaves),
            100.0 * leaves / total,
            ms(m.evaluation.as_secs_f64()),
        );
    }
    println!();
    println!("### Fastest leaf size per configuration");
    println!();
    println!(
        "| distribution | p | fastest | evaluate (ms) | at {LEAF_RULE_BASELINE} (ms) | \
         {LEAF_RULE_BASELINE} / fastest | evaluate at {} (ms) |",
        LEAF_SIZES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | --- |");
    for (d, p) in configurations(arguments) {
        let mine = configuration(runs, d, p);
        let time = |m: &Measurement| m.evaluation.as_secs_f64();
        let fastest = mine
            .iter()
            .min_by(|a, b| time(a).total_cmp(&time(b)))
            .expect("every size runs");
        let base = mine
            .iter()
            .find(|m| m.config.max_points_per_leaf == LEAF_RULE_BASELINE)
            .expect("the baseline size runs");
        println!(
            "| {} | {p} | {} | {} | {} | {:.2} | {} |",
            d.name(),
            fastest.config.max_points_per_leaf,
            ms(time(fastest)),
            ms(time(base)),
            time(base) / time(fastest),
            mine.iter()
                .map(|m| ms(time(m)))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

/// The device leaf-size rule over the study.
fn print_leaf_rule(runs: &[Measurement], arguments: &Arguments) {
    let configurations = configurations(arguments);
    let per_size: Vec<Vec<&Measurement>> = (0..LEAF_SIZES.len())
        .map(|i| {
            configurations
                .iter()
                .map(|&(d, p)| configuration(runs, d, p)[i])
                .collect()
        })
        .collect();
    let times: Vec<Vec<f64>> = per_size
        .iter()
        .map(|v| v.iter().map(|m| m.evaluation.as_secs_f64()).collect())
        .collect();
    let errors: Vec<Vec<(f64, f64)>> = per_size
        .iter()
        .map(|v| {
            v.iter()
                .map(|m| (m.run.potential.l2, m.run.gradient.l2))
                .collect()
        })
        .collect();
    let choice = leaf_size_rule(&LEAF_SIZES, &times, &errors);
    println!();
    println!("### The device leaf-size rule (T13)");
    println!();
    println!(
        "Geometric mean of the {} evaluation time over the {} configurations (cube and \
         Plummer, p = {:?}, {}). The fastest size replaces {LEAF_RULE_BASELINE} only if it is \
         at least {:.0}% faster by this measure and no φ or ∇φ L2 error at it is more than \
         {:.0}% worse than at {LEAF_RULE_BASELINE}.",
        arguments.backend,
        configurations.len(),
        arguments.leaf_degrees,
        arguments.precision.name(),
        100.0 * (LEAF_RULE_GAIN - 1.0),
        100.0 * (LEAF_RULE_ERROR_GROWTH - 1.0)
    );
    println!();
    println!("| max points per leaf | geometric mean (ms) | relative to {LEAF_RULE_BASELINE} |");
    println!("| ---: | ---: | ---: |");
    let base = LEAF_SIZES
        .iter()
        .position(|&s| s == LEAF_RULE_BASELINE)
        .expect("the baseline is a study size");
    for (size, g) in LEAF_SIZES.iter().zip(&choice.geometric_means) {
        println!(
            "| {size} | {} | {:.3} |",
            ms(*g),
            g / choice.geometric_means[base]
        );
    }
    println!();
    println!(
        "Fastest: {} ({:.3}x faster than {LEAF_RULE_BASELINE}); largest error growth there: \
         {:.3}x. **The rule picks {}.**",
        choice.fastest, choice.gain, choice.error_growth, choice.chosen
    );
}

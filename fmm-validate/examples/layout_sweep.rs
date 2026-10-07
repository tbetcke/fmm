//! The layout sweep of the device path (Phase 4S T7, C4S.7): every layout candidate of the
//! device kernels, measured on the level calls of an FMM's own plan, so that a backend's
//! defaults and tuner candidates can be chosen by measurement. Prints Markdown on stdout
//! and progress on stderr. Feature `gpu` (with a backend).
//!
//! The problem is the uniform cube of the C3.3 workload (`fmm_accuracy::Config::c33`):
//! N points (`--n`, default 10⁶), sources equal to targets, `max_level` 16, 64 points per
//! leaf, gradients on, in `--precision` (f32 or f64) at each of `--degrees`. Parts
//! (`--part`, default all):
//!
//! - **tuner**: the FMM built with a fresh tuning cache, a budget of `--budget` seconds
//!   (default 1,200) and a [`TuningHook`] that offers every decision a wide candidate set
//!   on top of the tuner's own, so that `nd_fmm_exec::tune` times them all on the FMM's
//!   level calls exactly as it times its own candidates (a warm-up launch, then the
//!   median of five batches of at least 10 ms of launches queued between syncs; level
//!   calls of fewer than 512 pairs keep the static rule):
//!   - the GEMM of M2M, L2L and dense M2L per pair bucket: the hand-written kernel in
//!     `GemmLayout::Cube { rows, columns, per_unit }` for rows ∈ {8, 16, 32, 64, 128}
//!     (at most n rounded up to a power of two), columns ∈ {1, 2, 4, 8, 16, 32},
//!     per_unit ∈ {1, 2, 4, 8}, with 32 to 1,024 units, at most 8 rows per unit and at
//!     most 64 accumulators per unit, each with the device's default chunk budget
//!     (`nd_fmm_kernels::translate::default_scratch_bytes`); and the backend's default
//!     layout with the other chunk budgets of 128 MB, 512 MB, 1 GB, 2 GB and 4 GB. Each
//!     time is the whole level call (gather, GEMM, reduction or scatter-add), reported
//!     with its rate in GFLOP/s (2 n² flops per pair) against the device's peak
//!     (`nd_fmm_validate::peaks`). A GEMM decision does not time a candidate whose chunks
//!     overrun the operator's scratch (sized for the tuner's own candidates); the strategy
//!     decision, which times the largest V level, gives it a scratch of its own;
//!   - the P2P layout: the cube layout with 32, 64, 128 and 256 units, the plane layout
//!     with 2, 4 and 8 planes per cube;
//!
//!   then the evaluation time (the median of five after one warm-up) of the tuned FMM
//!   against the static rule's.
//! - **kinds**: the FMM built once per variant of a builder setting, under the static
//!   rule otherwise, its time per operator kind from `KindTiming::Device` (CUDA events,
//!   a breakdown on CUDA; on Metal `Synchronous`, the only breakdown there; Phase 4S T5),
//!   the median over `--repeats` evaluations (default 5) after two warm-up evaluations,
//!   and the evaluation's wall time from a build without kind timings:
//!   - the leaf layout (`DeviceLeafLayout::Cube { units, tile }`, units ∈ {32, 64, 128,
//!     256}, tile ∈ {8, 16, 32, 64, 128} with tile ≤ units, where the shared memory holds
//!     the tile), timing P2M, P2L, L2P and M2P;
//!   - the P2P layout (as in **tuner**), timing P2P;
//!   - the chunk budget (`device_scratch_budget`: the device's default, then the others
//!     of 128 MB, 512 MB, 1 GB, 2 GB and 4 GB), timing M2M, L2L and M2L.
//! - **overheads**: on the device alone (`nd_fmm_kernels::Device`):
//!   - the cost of a launch: 10,000 launches of `movement::zero` over 256 values queued,
//!     then one sync; the host's enqueue time per launch and the whole per launch;
//!   - the cost of a sync: `Device::sync` on an idle device, and one launch followed by a
//!     sync, the median of 1,000;
//!   - the elementwise kernels: `zero` and `gather_columns` (n = 81, identity indices)
//!     over 2²⁰ to 2²⁸ values in the run's precision, GB/s against the device's
//!     bandwidth. A GPU elementwise launch has 256 units per cube and at most 65,535
//!     cubes (`GPU_MAX_CUBES` of `nd_fmm_kernels::device`): beyond 2²⁴ values its units
//!     stride, so the rows past 2²⁴ show what the cap costs.
//!
//! Nothing is asserted; every time is measured on this machine and reported.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features cuda --example layout_sweep -- \
//!     --device cuda --tuning-cache DIR [--precision f64] [--degrees 3,6,8] [--n 1000000] \
//!     [--part tuner|kinds|overheads|all] [--table-cache DIR]
//! ```
//!
//! On locust (Phase 4S, by hand): after `tools/gh200/sync.sh`, through
//! `tools/gh200/remote.sh`, the GPU otherwise idle (docs/phase4s/README.md, "Timing").
//! `--tuning-cache DIR` is needed by **tuner**: each (precision, p) gets the subdirectory
//! `DIR/sweep-<backend>-<precision>-p<pp>`, emptied first. `--table-cache DIR` loads and
//! stores the tables. `--threads n` sets the FMM's threads (the host-fallback kinds only;
//! default 1). The CPU runtime runs every part as a smoke test: its timings say nothing
//! about a GPU. Metal needs a process with GPU access (outside the macOS sandbox). It
//! initialises MPI at `Threading::Funneled` and runs on one rank.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{
    Backend, DeviceLeafLayout, DeviceP2pLayout, Fmm, FmmBuilder, KindTiming, OperatorKind,
};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tune::{Candidate, Decision, GemmChoice, Timing, TuningHook, TuningReport};
use nd_fmm_kernels::p2p::P2pLayout;
use nd_fmm_kernels::translate::{GemmLayout, default_scratch_bytes};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, DeviceInfo, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_validate::bench::{cores, cpu_model, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};
use nd_fmm_validate::peaks::{self, Peaks};

/// The parts of the sweep (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    Tuner,
    Kinds,
    Overheads,
}

/// The command line.
struct Arguments {
    device: String,
    precision: Option<String>,
    degrees: Option<Vec<usize>>,
    n: usize,
    parts: Vec<Part>,
    tuning_cache: Option<PathBuf>,
    table_cache: Option<PathBuf>,
    budget: f64,
    repeats: usize,
    threads: usize,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: layout_sweep --device cpu|metal|cuda [--precision f32|f64] \
             [--degrees p,p,...] [--n N] [--part tuner|kinds|overheads|all] \
             [--tuning-cache DIR] [--table-cache DIR] [--budget s] [--repeats r] \
             [--threads n]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        precision: None,
        degrees: None,
        n: 1_000_000,
        parts: vec![Part::Tuner, Part::Kinds, Part::Overheads],
        tuning_cache: None,
        table_cache: None,
        budget: 1200.0,
        repeats: 5,
        threads: 1,
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
            ("--n", [value, tail @ ..]) => {
                parsed.n = value
                    .parse::<f64>()
                    .ok()
                    .filter(|n| *n >= 1.0)
                    .unwrap_or_else(|| usage()) as usize;
                tail
            }
            ("--part", [value, tail @ ..]) => {
                parsed.parts = match value.as_str() {
                    "tuner" => vec![Part::Tuner],
                    "kinds" => vec![Part::Kinds],
                    "overheads" => vec![Part::Overheads],
                    "all" => vec![Part::Tuner, Part::Kinds, Part::Overheads],
                    _ => usage(),
                };
                tail
            }
            ("--tuning-cache", [value, tail @ ..]) => {
                parsed.tuning_cache = Some(PathBuf::from(value));
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
            ("--repeats", [value, tail @ ..]) => {
                parsed.repeats = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            ("--threads", [value, tail @ ..]) => {
                parsed.threads = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty() || parsed.repeats == 0 || parsed.threads == 0 {
        usage();
    }
    if parsed.parts.contains(&Part::Tuner) && parsed.tuning_cache.is_none() {
        eprintln!("layout_sweep: the tuner part needs --tuning-cache DIR");
        std::process::exit(2);
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
    let precision = match arguments.precision.as_deref() {
        None | Some("f32") => Precision::F32,
        Some("f64") => Precision::F64,
        Some(other) => {
            eprintln!("--precision {other}: expected f32 or f64");
            std::process::exit(2);
        }
    };
    if !info.supports(precision) {
        eprintln!(
            "--precision {precision}: {} does no arithmetic in it",
            info.name
        );
        std::process::exit(1);
    }
    let degrees = arguments.degrees.clone().unwrap_or(match precision {
        Precision::F32 => vec![3, 6, 8],
        Precision::F64 => vec![3, 6, 8, 12],
    });
    let peak = Peaks::of_info(&info);
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let comm = universe.world();

    println!("# Device layout sweep (Phase 4S T7, C4S.7)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {info} |");
    println!(
        "| device limits | plane {:?}, shared memory {} B per cube, {} units per cube, cube \
         counts {:?} |",
        info.plane_size, info.max_shared_memory, info.max_units_per_cube, info.max_cube_count
    );
    println!("| precision | {precision} |");
    println!(
        "| problem | the uniform cube of C3.3, N = {}, `max_level` 16, 64 points per leaf, \
         gradients on |",
        arguments.n
    );
    match peak {
        Some(peak) => {
            println!(
                "| peak | {} GFLOP/s ({}) |",
                peak.gflops(peak_precision(precision))
                    .map_or("none".to_owned(), |g| g.to_string()),
                peak.source(peak_precision(precision))
            );
            println!(
                "| bandwidth | {} GB/s ({}) |",
                peak.bandwidth_gbs, peak.bandwidth_source
            );
        }
        None => println!("| peak | unknown for this device |"),
    }
    println!(
        "| repeats | tuner: median of 5 batches of >= 10 ms; kinds: median of {} \
         evaluations after 2 warm-ups; overheads: as each table says |",
        arguments.repeats
    );
    if backend == Backend::Cpu {
        println!(
            "| note | the CubeCL CPU runtime is the correctness backend: these timings \
             exercise the sweep and say nothing about a GPU |"
        );
    }
    println!();

    let config = Config {
        n: arguments.n,
        sampled: 1,
        charge_vectors: 1,
        ..Config::c33(Distribution::Cube)
    };
    eprintln!("cube: drawing N = {}", config.n);
    let problem = Problem::new(&config);
    let sweep = Sweep {
        backend,
        info: &info,
        precision,
        peak,
        config: &config,
        problem: &problem,
        arguments: &arguments,
        comm: &comm,
    };
    for &p in &degrees {
        for part in &arguments.parts {
            match (part, precision) {
                (Part::Tuner, Precision::F32) => sweep.tuner::<f32>(p),
                (Part::Tuner, Precision::F64) => sweep.tuner::<f64>(p),
                (Part::Kinds, Precision::F32) => sweep.kinds::<f32>(p),
                (Part::Kinds, Precision::F64) => sweep.kinds::<f64>(p),
                (Part::Overheads, _) => {}
            }
        }
    }
    if arguments.parts.contains(&Part::Overheads) {
        let kind = BackendKind::from_name(&arguments.device).expect("a parsed backend");
        let mut device = Device::open(kind).expect("the device opens again");
        match precision {
            Precision::F32 => overheads::<f32>(&mut device, peak),
            Precision::F64 => overheads::<f64>(&mut device, peak),
        }
    }
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

/// The precision of the peak table.
fn peak_precision(precision: Precision) -> peaks::Precision {
    match precision {
        Precision::F32 => peaks::Precision::F32,
        Precision::F64 => peaks::Precision::F64,
    }
}

/// The hand-written GEMM layouts the tuner part offers for tables of order n on `info`
/// (module documentation).
fn gemm_layouts(info: &DeviceInfo, n: usize) -> Vec<GemmLayout> {
    let mut out = Vec::new();
    for rows in [8u32, 16, 32, 64, 128] {
        if rows as usize > n.next_power_of_two().max(8) {
            continue;
        }
        let tm = n.div_ceil(rows as usize);
        if tm > 8 {
            continue;
        }
        for columns in [1u32, 2, 4, 8, 16, 32] {
            let units = rows * columns;
            if !(32..=1024).contains(&units) || units > info.max_units_per_cube {
                continue;
            }
            for per_unit in [1u32, 2, 4, 8] {
                if tm * per_unit as usize > 64 {
                    continue;
                }
                out.push(GemmLayout::Cube {
                    rows,
                    columns,
                    per_unit,
                });
            }
        }
    }
    out
}

/// The P2P layouts of the sweep.
fn p2p_layouts() -> Vec<P2pLayout> {
    [32, 64, 128, 256]
        .map(|units| P2pLayout::Cube { units })
        .into_iter()
        .chain([2, 4, 8].map(|planes| P2pLayout::Plane { planes }))
        .collect()
}

/// The candidates of a decision table shown, fastest first (the static rule's always).
const SHOWN: usize = 16;

/// The chunk budgets of the sweep; the device's default is left out of the offers.
const BUDGETS: [u64; 5] = [128 << 20, 512 << 20, 1 << 30, 2 << 30, 4 << 30];

/// One sweep: the problem and the settings shared by every part.
struct Sweep<'a> {
    backend: Backend,
    info: &'a DeviceInfo,
    precision: Precision,
    peak: Option<Peaks>,
    config: &'a Config,
    problem: &'a Problem,
    arguments: &'a Arguments,
    comm: &'a SimpleCommunicator,
}

impl Sweep<'_> {
    /// The builder of the sweep's FMM at degree p, under the static rule.
    fn builder<T>(&self, p: usize) -> FmmBuilder<T> {
        let mut builder = FmmBuilder::<T>::new(p)
            .gradients(true)
            .max_level(self.config.max_level)
            .max_points_per_leaf(self.config.max_points_per_leaf)
            .threads(self.arguments.threads)
            .backend(self.backend);
        if let Some(tables) = &self.arguments.table_cache {
            builder = builder.table_cache(tables);
        }
        builder
    }

    /// The charges of the first vector in T.
    fn charges<T: RealScalar>(&self) -> Vec<T> {
        self.problem.charges[0]
            .iter()
            .map(|&q| T::from_f64(q))
            .collect()
    }

    /// The tuner part at degree p (module documentation).
    fn tuner<T>(&self, p: usize)
    where
        T: Stored + SimdScalar + Equivalence + Default + RealScalar,
    {
        let n = (p + 1) * (p + 1);
        let dir = self
            .arguments
            .tuning_cache
            .as_ref()
            .expect("checked with the arguments")
            .join(format!("sweep-{}-{}-p{p:02}", self.backend, self.precision));
        let _ = std::fs::remove_dir_all(&dir);
        let info = self.info.clone();
        // The GEMM choices, offered to the GEMM decisions and, as `Dense`, to the strategy
        // decision, which times the M2L GEMM of the largest V level (that bucket's GEMM
        // decision takes those times).
        let choices = move |info: &DeviceInfo| {
            let budget = default_scratch_bytes(info);
            let mut out: Vec<GemmChoice> = gemm_layouts(info, n)
                .into_iter()
                .map(|layout| GemmChoice {
                    budget,
                    ..GemmChoice::hand_written(layout)
                })
                .collect();
            let default = GemmChoice {
                budget,
                ..GemmChoice::hand_written(GemmLayout::default_for(info, n))
            };
            out.extend(
                BUDGETS
                    .into_iter()
                    .filter(|&b| b != budget)
                    .map(|budget| GemmChoice { budget, ..default }),
            );
            out
        };
        let hook = TuningHook::new().offer(move |decision| match decision {
            Decision::Gemm { .. } => choices(&info).into_iter().map(Candidate::Gemm).collect(),
            Decision::Strategy => choices(&info).into_iter().map(Candidate::Dense).collect(),
            Decision::P2p { .. } => p2p_layouts().into_iter().map(Candidate::P2p).collect(),
        });
        eprintln!("tuner, p = {p}: tuning");
        let points = &self.problem.points;
        let start = Instant::now();
        let mut tuned = self
            .builder::<T>(p)
            .tuning_cache(&dir)
            .tuning_budget(Duration::from_secs_f64(self.arguments.budget))
            .tuning_hook(hook)
            .build(points, points, self.comm)
            .unwrap_or_else(|error| panic!("the tuned FMM does not build: {error}"));
        let build = start.elapsed();
        let report = tuned
            .device_report()
            .and_then(|r| r.tuning.clone())
            .expect("a device build reports its tuning");
        self.print_tuning(p, &report, build);
        eprintln!("tuner, p = {p}: evaluating");
        let charges = self.charges::<T>();
        let mut fixed = self
            .builder::<T>(p)
            .build(points, points, self.comm)
            .expect("the static-rule FMM builds");
        let static_ms = median_evaluation(&mut fixed, &charges, 5) * 1e3;
        let tuned_ms = median_evaluation(&mut tuned, &charges, 5) * 1e3;
        println!(
            "Evaluation (median of 5 after one warm-up): static rule {static_ms:.3} ms, tuned \
             {tuned_ms:.3} ms ({:.2}× faster).",
            static_ms / tuned_ms
        );
        println!();
    }

    /// The tuning tables, with the GEMM rates.
    fn print_tuning(&self, p: usize, report: &TuningReport, build: Duration) {
        let n = ((p + 1) * (p + 1)) as f64;
        let peak = self
            .peak
            .and_then(|peak| peak.gflops(peak_precision(self.precision)));
        println!("## Tuner, {}, p = {p}", self.precision);
        println!();
        println!(
            "Tuning {:.1} s of a {:.0} s budget{}; build {:.1} s.",
            report.time.as_secs_f64(),
            report.budget.as_secs_f64(),
            if report.deadline_hit {
                ", deadline reached"
            } else {
                ""
            },
            build.as_secs_f64(),
        );
        println!();
        for d in &report.decisions {
            let level = d.level.map_or("–".to_owned(), |l| l.to_string());
            println!("### {} (level {level}, {} pairs)", d.decision, d.pairs);
            println!();
            if d.times.is_empty() {
                println!(
                    "Not timed: {} ({}{}).",
                    d.choice,
                    d.source,
                    d.note
                        .as_deref()
                        .map_or(String::new(), |n| format!("; {n}"))
                );
                println!();
                continue;
            }
            let gemm = matches!(d.decision, Decision::Gemm { .. } | Decision::Strategy);
            println!("| candidate | time µs | GFLOP/s | % peak | / best | |");
            println!("| --- | ---: | ---: | ---: | ---: | --- |");
            let best = d.best();
            let first = d.times[0].0;
            // Measured candidates fastest first, then the rest in registration order.
            let mut order: Vec<usize> = (0..d.times.len()).collect();
            order.sort_by_key(|&i| match d.times[i].1 {
                Timing::Measured(t) => (0, t),
                _ => (1, Duration::ZERO),
            });
            for (shown, &i) in order.iter().enumerate() {
                let (candidate, timing) = &d.times[i];
                // Long tables: the fastest 16, then only the static rule's.
                if shown >= SHOWN && *candidate != first {
                    continue;
                }
                let mut marks = Vec::new();
                if *candidate == d.choice {
                    marks.push(format!("**chosen** ({})", d.source));
                }
                if *candidate == first {
                    marks.push("static rule".to_owned());
                }
                match timing {
                    Timing::Measured(t) => {
                        let seconds = t.as_secs_f64();
                        let rate = 2.0 * n * n * d.pairs as f64 / seconds / 1e9;
                        let dense = !matches!(candidate, Candidate::Rotation | Candidate::P2p(_));
                        let (rate, share) = if gemm && dense {
                            (format!("{rate:.0}"), peaks::percent(rate, peak))
                        } else {
                            ("–".to_owned(), "–".to_owned())
                        };
                        println!(
                            "| {candidate} | {:.1} | {rate} | {share} | {:.2} | {} |",
                            seconds * 1e6,
                            best.map_or(1.0, |b| seconds / b.as_secs_f64()),
                            marks.join(", ")
                        );
                    }
                    other => {
                        marks.push(other.to_string());
                        println!("| {candidate} | – | – | – | – | {} |", marks.join(", "));
                    }
                }
            }
            if d.times.len() > SHOWN + 1 {
                println!(
                    "\n{} candidates in all; the fastest {SHOWN} and the static rule's shown.",
                    d.times.len()
                );
            }
            println!();
        }
    }

    /// The kinds part at degree p (module documentation).
    fn kinds<T>(&self, p: usize)
    where
        T: Stored + SimdScalar + Equivalence + Default + RealScalar,
    {
        let n = (p + 1) * (p + 1);
        let value = match self.precision {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        let mode = if self.backend == Backend::Metal || self.backend == Backend::Cpu {
            KindTiming::Synchronous
        } else {
            KindTiming::Device
        };
        let leaf_kinds = [
            OperatorKind::P2m,
            OperatorKind::P2l,
            OperatorKind::L2p,
            OperatorKind::M2p,
        ];
        let mut leaf: Vec<(String, FmmBuilder<T>)> =
            vec![("default".to_owned(), self.builder::<T>(p))];
        for units in [32u32, 64, 128, 256] {
            for tile in [8u32, 16, 32, 64, 128] {
                let shared = (tile as usize * (n + 1)).max(n) * value;
                if tile > units
                    || units > self.info.max_units_per_cube
                    || shared > self.info.max_shared_memory
                {
                    continue;
                }
                leaf.push((
                    format!("cube ({units} units, tile {tile})"),
                    self.builder::<T>(p)
                        .device_leaf_layout(DeviceLeafLayout::Cube { units, tile }),
                ));
            }
        }
        self.kinds_table(p, "leaf layouts", &leaf_kinds, leaf, mode);

        let mut p2p: Vec<(String, FmmBuilder<T>)> =
            vec![("default".to_owned(), self.builder::<T>(p))];
        for layout in p2p_layouts() {
            let setting = match layout {
                P2pLayout::Cube { units } => DeviceP2pLayout::Cube(units),
                P2pLayout::Plane { planes } => DeviceP2pLayout::Plane(planes),
                P2pLayout::Cpu { .. } => continue,
            };
            if layout.check(self.info, self.precision).is_err() {
                continue;
            }
            p2p.push((
                layout.to_string(),
                self.builder::<T>(p).device_p2p_layout(setting),
            ));
        }
        self.kinds_table(p, "P2P layouts", &[OperatorKind::P2p], p2p, mode);

        let default_budget = default_scratch_bytes(self.info);
        let mut budgets: Vec<(String, FmmBuilder<T>)> = vec![(
            format!("{} MB (default)", default_budget >> 20),
            self.builder::<T>(p),
        )];
        for budget in BUDGETS.into_iter().filter(|&b| b != default_budget) {
            budgets.push((
                format!("{} MB", budget >> 20),
                self.builder::<T>(p).device_scratch_budget(budget),
            ));
        }
        self.kinds_table(
            p,
            "chunk budgets",
            &[OperatorKind::M2m, OperatorKind::L2l, OperatorKind::M2l],
            budgets,
            mode,
        );
    }

    /// One table of the kinds part: per variant, the median time of `kinds` and the
    /// evaluation.
    fn kinds_table<T>(
        &self,
        p: usize,
        what: &str,
        kinds: &[OperatorKind],
        variants: Vec<(String, FmmBuilder<T>)>,
        mode: KindTiming,
    ) where
        T: Stored + SimdScalar + Equivalence + Default + RealScalar,
    {
        let points = &self.problem.points;
        let charges = self.charges::<T>();
        println!(
            "## {what}, {}, p = {p} (kind timings {mode:?})",
            self.precision
        );
        println!();
        let names: Vec<String> = kinds.iter().map(|k| format!("{k} ms")).collect();
        println!(
            "| variant | {} | kinds ms | evaluation ms | / default |",
            names.join(" | ")
        );
        println!(
            "| --- | {} ---: | ---: | ---: |",
            "---: | ".repeat(kinds.len())
        );
        let mut default_ms = None;
        for (name, builder) in variants {
            eprintln!("kinds, p = {p}: {what}: {name}");
            let mut plain = match builder.clone().build(points, points, self.comm) {
                Ok(fmm) => fmm,
                Err(error) => {
                    println!("| {name} | refused: {error} | | | |");
                    continue;
                }
            };
            let evaluation = median_evaluation(&mut plain, &charges, self.arguments.repeats);
            drop(plain);
            let mut timed = builder
                .kind_timings(mode)
                .build(points, points, self.comm)
                .expect("the variant builds with kind timings");
            for _ in 0..2 {
                timed.evaluate(&charges).expect("it evaluates");
            }
            let mut per_kind: Vec<Vec<f64>> = vec![Vec::new(); kinds.len()];
            for _ in 0..self.arguments.repeats {
                let output = timed.evaluate(&charges).expect("it evaluates");
                if let Some(times) = output.timings.kinds {
                    for (i, kind) in kinds.iter().enumerate() {
                        per_kind[i].push(times.get(*kind).total().as_secs_f64());
                    }
                }
            }
            let medians: Vec<f64> = per_kind.iter_mut().map(|t| median(t)).collect();
            let sum: f64 = medians.iter().sum();
            let default = *default_ms.get_or_insert(sum);
            println!(
                "| {name} | {} | {:.3} | {:.3} | {:.2} |",
                medians
                    .iter()
                    .map(|t| format!("{:.3}", t * 1e3))
                    .collect::<Vec<_>>()
                    .join(" | "),
                sum * 1e3,
                evaluation * 1e3,
                if default > 0.0 { sum / default } else { 1.0 },
            );
        }
        println!();
    }
}

/// The median of `values`, NaN if empty.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The median wall time of `repeats` evaluations after one warm-up, in seconds.
fn median_evaluation<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &mut Fmm<'_, T>,
    charges: &[T],
    repeats: usize,
) -> f64 {
    fmm.evaluate(charges).expect("it evaluates");
    let mut times: Vec<f64> = (0..repeats.max(1))
        .map(|_| {
            let start = Instant::now();
            fmm.evaluate(charges).expect("it evaluates");
            start.elapsed().as_secs_f64()
        })
        .collect();
    median(&mut times)
}

/// The overheads part (module documentation).
fn overheads<T: DeviceFloat + RealScalar>(device: &mut Device, peak: Option<Peaks>) {
    let bandwidth = peak.map(|p| p.bandwidth_gbs);
    println!("## Launches and syncs ({})", device.backend());
    println!();
    let mut tiny = device.alloc::<T>(256).expect("a small buffer");
    for _ in 0..100 {
        nd_fmm_kernels::movement::zero(device, tiny.as_slice_mut()).unwrap();
    }
    device.sync().unwrap();
    let launches = 10_000;
    let mut enqueue = Vec::new();
    let mut whole = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..launches {
            nd_fmm_kernels::movement::zero(device, tiny.as_slice_mut()).unwrap();
        }
        enqueue.push(start.elapsed().as_secs_f64() / launches as f64);
        device.sync().unwrap();
        whole.push(start.elapsed().as_secs_f64() / launches as f64);
    }
    let mut idle = Vec::new();
    let mut round = Vec::new();
    for _ in 0..1000 {
        let start = Instant::now();
        device.sync().unwrap();
        idle.push(start.elapsed().as_secs_f64());
        let start = Instant::now();
        nd_fmm_kernels::movement::zero(device, tiny.as_slice_mut()).unwrap();
        device.sync().unwrap();
        round.push(start.elapsed().as_secs_f64());
    }
    println!("| measure | µs (median) |");
    println!("| --- | ---: |");
    println!(
        "| enqueue one launch (10,000 queued, of 5 batches) | {:.2} |",
        median(&mut enqueue) * 1e6
    );
    println!(
        "| one launch, queued and run (10,000 then one sync, of 5 batches) | {:.2} |",
        median(&mut whole) * 1e6
    );
    println!(
        "| a sync on an idle device (of 1,000) | {:.2} |",
        median(&mut idle) * 1e6
    );
    println!(
        "| one launch and a sync (of 1,000) | {:.2} |",
        median(&mut round) * 1e6
    );
    println!();
    drop(tiny);

    let value = size_of::<T>();
    println!(
        "## Elementwise kernels ({}, {} B values; 256 units per cube, at most 65,535 cubes)",
        device.backend(),
        value
    );
    println!();
    println!("| kernel | values | cubes | units stride | µs | GB/s | % bandwidth |");
    println!("| --- | ---: | ---: | --- | ---: | ---: | ---: |");
    let n = 81usize;
    for exponent in [20u32, 22, 24, 25, 26, 27, 28] {
        let len = (1usize << exponent).next_multiple_of(n);
        let Ok(mut y) = device.alloc::<T>(len) else {
            println!("| zero | {len} | refused: no memory | | | | |");
            continue;
        };
        // The grid of `nd_fmm_kernels::device::Grid::elementwise` on a GPU.
        let cubes = len.div_ceil(ELEMENTWISE_UNITS).min(ELEMENTWISE_MAX_CUBES);
        let strides = cubes * ELEMENTWISE_UNITS < len;
        let (cubes, strides) = if device.backend().is_gpu() {
            (cubes.to_string(), if strides { "yes" } else { "no" })
        } else {
            ("–".to_owned(), "–")
        };
        let seconds = time_device(device, |d| {
            nd_fmm_kernels::movement::zero(d, y.as_slice_mut()).unwrap();
        });
        let bytes = (len * value) as f64;
        println!(
            "| zero | {len} | {} | {} | {:.1} | {:.0} | {} |",
            cubes,
            strides,
            seconds * 1e6,
            bytes / seconds / 1e9,
            peaks::percent(bytes / seconds / 1e9, bandwidth)
        );
        let columns = len / n;
        let identity: Vec<u32> = (0..columns as u32).collect();
        let indices = device
            .upload_indices(&identity)
            .expect("the indices upload");
        let x = device.alloc::<T>(len).expect("the input");
        let seconds = time_device(device, |d| {
            nd_fmm_kernels::movement::gather_columns(
                d,
                n,
                x.as_slice(),
                indices.as_slice(),
                y.as_slice_mut(),
            )
            .unwrap();
        });
        // Read x and the indices, write y.
        let bytes = (2 * len * value + 4 * columns) as f64;
        println!(
            "| gather_columns (n = {n}) | {len} | {} | {} | {:.1} | {:.0} | {} |",
            cubes,
            strides,
            seconds * 1e6,
            bytes / seconds / 1e9,
            peaks::percent(bytes / seconds / 1e9, bandwidth)
        );
    }
    println!();
}

/// The units per cube of a GPU elementwise launch (`nd_fmm_kernels::device`, `GPU_UNITS`).
const ELEMENTWISE_UNITS: usize = 256;

/// The most cubes of a GPU elementwise launch (`nd_fmm_kernels::device`, `GPU_MAX_CUBES`).
const ELEMENTWISE_MAX_CUBES: usize = 65_535;

/// The median seconds of `run`, launches queued between syncs, after a warm-up launch
/// and a sync.
fn time_device(device: &mut Device, mut run: impl FnMut(&mut Device)) -> f64 {
    run(device);
    device.sync().unwrap();
    nd_fmm_validate::bench::median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            run(device);
        }
        device.sync().unwrap();
        start.elapsed()
    })
}

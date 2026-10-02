//! Calibration of the degree p against the accuracy of the complete FMM (C3.4), and a
//! leaf-size study. Prints Markdown on stdout and progress on stderr.
//!
//! For each of the four distributions (the uniform cube, the sphere surface, the
//! Plummer sphere and the Gaussian clusters): N = 10⁵ points, sources equal to targets,
//! eight charge vectors uniform in [−1, 1), an adaptive tree with `max_level` 16 and 64
//! points per leaf as the refinement target (the settings of T11, for every
//! distribution), the default strategy (`Auto`) and gradients on. The FMM runs in f64
//! for p = 1..=20 and in f32 for p = 1..=8, against the f64 direct sum at 1,000 sampled
//! targets, computed once per distribution and precision.
//!
//! The report has
//! - the calibration tables: for each target 10⁻ᵏ, k = 3..=12, the smallest p whose
//!   relative L2 error of φ (then of ∇φ) is below it, per distribution and as the worst
//!   of the four, and the f32 floor;
//! - per distribution, the tree and a table per precision of the errors, the build and
//!   the stage timings;
//! - the leaf-size study: p = 8 in f64 on the cube and the Plummer sphere with 16, 32,
//!   64, 128 and 256 points per leaf as the refinement target.
//!
//! Run in release mode, on one rank (the points are not redistributed until C5.1), with
//! `--threads n` rayon threads (default 1), `--n N` points (default 10⁵) and one BLAS
//! thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example calibrate -- --threads 8
//! ```
//!
//! The problems and the error measure are described in `nd_fmm_validate::calibration`
//! and `nd_fmm_validate::fmm_accuracy`. The errors are deterministic for the seed and
//! the same for every number of threads; the timings are wall times on this machine,
//! reported and never asserted. MPI is initialised with `Threading::Funneled`, and the
//! threading report (rayon threads, MPI level, BLAS variables;
//! `nd_fmm_exec::threading`) is printed with the results.

use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::calibration::{
    LEAF_SIZES, LEAF_STUDY_DEGREE, LEAF_STUDY_DISTRIBUTIONS, Measure, Precision, Reached,
    Reference, TARGET_EXPONENTS, config, floor, leaf_study, smallest_p, sweep, worst,
};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Run};

/// The command line: the threads and the number of points.
struct Arguments {
    threads: usize,
    n: usize,
}

/// Parses `--threads n` and `--n N`, in any order; exits with a message on anything
/// else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!("usage: calibrate [--threads n] [--n N], n >= 1, N >= 1000; got {args:?}");
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        threads: 1,
        n: 100_000,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match (flag.as_str(), value.parse::<usize>()) {
            ("--threads", Ok(n)) if n >= 1 => parsed.threads = n,
            ("--n", Ok(n)) if n >= 1000 => parsed.n = n,
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() {
        usage();
    }
    parsed
}

/// The sweeps of one distribution, in f64 and in f32.
struct Report {
    config: Config,
    reference: Reference,
    f64: Vec<Run>,
    f32: Vec<Run>,
}

impl Report {
    /// The sweep of `precision`.
    fn runs(&self, precision: Precision) -> &[Run] {
        match precision {
            Precision::F32 => &self.f32,
            Precision::F64 => &self.f64,
        }
    }
}

/// Runs `degrees` one at a time, with progress on stderr.
fn progress_sweep(
    config: &Config,
    reference: &Reference,
    precision: Precision,
    threads: usize,
    comm: &SimpleCommunicator,
) -> Vec<Run> {
    precision
        .degrees()
        .iter()
        .flat_map(|&p| {
            let start = Instant::now();
            let runs = sweep(config, reference, precision, &[p], threads, comm);
            eprintln!(
                "{} {} p = {p}: φ L2 {:.3e}, ∇φ L2 {:.3e} ({:.1} s)",
                config.distribution.name(),
                precision.name(),
                runs[0].potential.l2,
                runs[0].gradient.l2,
                start.elapsed().as_secs_f64()
            );
            runs
        })
        .collect()
}

fn main() {
    let Arguments { threads, n } = arguments();
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if threads > 1 && provided < Threading::Funneled {
        eprintln!("--threads {threads} needs MPI at Funneled; it provides {provided:?}");
        std::process::exit(2);
    }
    if comm.size() != 1 {
        eprintln!("calibrate runs on one rank; points are not redistributed until C5.1");
        std::process::exit(2);
    }

    let start = Instant::now();
    let reports: Vec<Report> = Distribution::ALL
        .iter()
        .map(|&d| {
            let config = config(d, n);
            let reference = Reference::new(&config);
            eprintln!(
                "{}: oracles in {:.1} s",
                d.name(),
                reference.time.as_secs_f64()
            );
            let f64 = progress_sweep(&config, &reference, Precision::F64, threads, &comm);
            let f32 = progress_sweep(&config, &reference, Precision::F32, threads, &comm);
            Report {
                config,
                reference,
                f64,
                f32,
            }
        })
        .collect();
    let sweep_time = start.elapsed();
    let start = Instant::now();
    let study: Vec<(Distribution, Vec<Run>)> = LEAF_STUDY_DISTRIBUTIONS
        .iter()
        .map(|&d| {
            let report = reports
                .iter()
                .find(|r| r.config.distribution == d)
                .expect("every distribution is swept");
            let runs = leaf_study(
                &report.config,
                &report.reference,
                LEAF_STUDY_DEGREE,
                &LEAF_SIZES,
                threads,
                &comm,
            );
            eprintln!("{}: leaf-size study done", d.name());
            (d, runs)
        })
        .collect();
    let study_time = start.elapsed();

    print_header(&reports, threads, (sweep_time, study_time));
    print_calibration(&reports);
    for report in &reports {
        print_report(report);
    }
    print_leaf_study(&study, &reports[0].config);
}

/// The problem, the error measure, the machine and the threading.
fn print_header(reports: &[Report], threads: usize, (sweep, study): (Duration, Duration)) {
    let config = &reports[0].config;
    println!("# Calibration of p against the accuracy of the FMM (C3.4)");
    println!();
    println!(
        "- Problems: N = {} points per distribution, sources equal to targets; {} charge \
         vectors uniform in [-1, 1) (seeds {:#x} + 1 + k); the domain from \
         `compute_global_bounding_box`; the default M2L strategy (`Auto`: `Dense` for \
         p <= 8, `Rotation` above); gradients on. Every distribution, the cube included, \
         has an adaptive tree with `max_level` {} and `max_points_per_leaf` {}. f64 at p \
         = {}..={}, f32 at p = {}..={}.",
        config.n,
        config.charge_vectors,
        config.seed,
        config.max_level,
        config.max_points_per_leaf,
        Precision::F64.degrees()[0],
        Precision::F64.degrees().last().expect("degrees"),
        Precision::F32.degrees()[0],
        Precision::F32.degrees().last().expect("degrees"),
    );
    println!(
        "- Error measure: relative L2 and max error of φ and of ∇φ (Euclidean norm per \
         target) at {} targets sampled with the seed of the points, against `direct_sum` \
         in f64 over all sources divided by 4π; for each charge vector, then the root \
         mean square over the vectors. \"range\" is the smallest and largest φ L2 error of \
         one vector. f32 runs use the charges rounded to f32, and the oracle on the \
         rounded charges. The oracles are computed once per distribution and precision.",
        config.sampled
    );
    println!(
        "- Machine: {}; {}; {}; one rank, {threads} thread{}. Wall time: sweeps {:.1} s, \
         leaf-size study {:.1} s; the oracles (one thread) {} s.",
        cpu_model(),
        cores(),
        target(),
        if threads == 1 { "" } else { "s" },
        sweep.as_secs_f64(),
        study.as_secs_f64(),
        reports
            .iter()
            .map(|r| format!(
                "{:.1} ({})",
                r.reference.time.as_secs_f64(),
                r.config.distribution.name()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let threading = &reports[0].f64[0].threading;
    let warnings = threading.warnings();
    println!(
        "- Threading: {threading}.{}",
        if warnings.is_empty() {
            String::new()
        } else {
            format!(
                " Warning: with {threads} rayon threads, set {} to 1 in the launching \
                 shell (no BLAS routine runs in a worker in Phase 3, so the results are \
                 unaffected).",
                warnings
                    .iter()
                    .map(|v| v.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
}

/// The calibration tables: per precision and measure, the smallest p for each target.
fn print_calibration(reports: &[Report]) {
    println!();
    println!("## Calibration");
    println!();
    println!(
        "The smallest p whose relative L2 error (root mean square over the charge vectors) \
         is below the target, per distribution and the worst of the four. \"> p\" means \
         no degree up to p reaches it. f32: the floor is the smallest error of the sweep, \
         at the p given, next to the f64 error at that p; where the two agree, rounding \
         has not yet set the floor and truncation still dominates."
    );
    for precision in [Precision::F64, Precision::F32] {
        for measure in [Measure::Potential, Measure::Gradient] {
            println!();
            println!("### {}, {}", precision.name(), measure.symbol());
            println!();
            print!("| target |");
            for r in reports {
                print!(" {} |", r.config.distribution.name());
            }
            println!(" worst |");
            println!("|---:|{}---:|", "---:|".repeat(reports.len()));
            for k in TARGET_EXPONENTS {
                let reached: Vec<Reached> = reports
                    .iter()
                    .map(|r| smallest_p(r.runs(precision), measure, 10f64.powi(-k)))
                    .collect();
                print!("| 1e-{k} |");
                for p in &reached {
                    print!(" {p} |");
                }
                println!(" {} |", worst(&reached));
            }
            if precision == Precision::F32 {
                print!("| floor |");
                let floors: Vec<(f64, usize, f64)> = reports
                    .iter()
                    .map(|r| {
                        let (e, p) = floor(&r.f32, measure);
                        let f64_at_p = r.f64.iter().find(|q| q.p == p).expect("f64 sweeps p");
                        (e, p, measure.of(f64_at_p))
                    })
                    .collect();
                for (e, p, e64) in &floors {
                    print!(" {e:.2e} (p = {p}; f64 {e64:.2e}) |");
                }
                let (e, p, e64) = floors
                    .iter()
                    .copied()
                    .max_by(|a, b| a.0.total_cmp(&b.0))
                    .expect("four distributions");
                println!(" {e:.2e} (p = {p}; f64 {e64:.2e}) |");
            }
        }
    }
}

/// Milliseconds with one decimal.
fn ms(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64() * 1e3)
}

/// The tree line of a run: levels, leaves, leaf levels, points per leaf, list sizes.
fn tree_row(r: &Run) -> String {
    let levels: Vec<usize> = (0..r.leaf_levels.len())
        .filter(|&l| r.leaf_levels[l] > 0)
        .collect();
    format!(
        "{} | {} | {} – {} | {} / {:.1} / {} | {} | {} | {} | {}",
        r.nlevels,
        r.nleaves,
        levels.first().copied().unwrap_or(0),
        levels.last().copied().unwrap_or(0),
        r.points_per_leaf.0,
        r.points_per_leaf.1,
        r.points_per_leaf.2,
        r.lists.u,
        r.lists.v,
        r.lists.w,
        r.lists.x
    )
}

/// The tree and the sweep tables of one distribution.
fn print_report(report: &Report) {
    let config = &report.config;
    let tree = &report.f64[0];
    println!();
    println!("## {}", config.distribution.name());
    println!();
    println!(
        "{}; N = {}, `max_level` {}, `max_points_per_leaf` {}, seed {:#x}.",
        config.distribution.description(),
        config.n,
        config.max_level,
        config.max_points_per_leaf,
        config.seed
    );
    println!();
    println!(
        "| levels | leaves | leaf levels (min – max) | points per leaf (min / mean / max) | U pairs | V pairs | W pairs | X pairs |"
    );
    println!("|---:|---:|---|---:|---:|---:|---:|---:|");
    println!("| {} |", tree_row(tree));
    println!();
    println!(
        "Leaves per level: {}.",
        (0..tree.leaf_levels.len())
            .filter(|&l| tree.leaf_levels[l] > 0)
            .map(|l| format!("{l}: {}", tree.leaf_levels[l]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for precision in [Precision::F64, Precision::F32] {
        println!();
        println!("### {} ({})", config.distribution.name(), precision.name());
        println!();
        println!(
            "Wall time in ms. Build: once per row (tables: built, no cache). Evaluate: the \
             mean over the {} charge vectors; upward = P2M and M2M, downward = L2L, M2L and \
             P2L, leaves = L2P, M2P and P2P, other = loading the charges, the exchanges and \
             the output. Total = build + one evaluate.",
            config.charge_vectors
        );
        println!();
        println!(
            "| p | strategy | φ L2 | φ L2 range | φ max | ∇φ L2 | ∇φ max | build | tables | upward | downward | leaves | other | evaluate | total |"
        );
        println!("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for r in report.runs(precision) {
            let (b, s) = (&r.build, &r.stages);
            let upward = s.upward_local + s.upward_global;
            let other = s.load + s.exchange_sources + s.exchange_multipoles + s.output;
            println!(
                "| {} | {:?} | {:.3e} | {:.2e} – {:.2e} | {:.3e} | {:.3e} | {:.3e} | {} | {} | {} | {} | {} | {} | {} | {} |",
                r.p,
                r.strategy,
                r.potential.l2,
                r.potential_l2_range.0,
                r.potential_l2_range.1,
                r.potential.max,
                r.gradient.l2,
                r.gradient.max,
                ms(b.total()),
                ms(b.tables),
                ms(upward),
                ms(s.downward),
                ms(s.evaluate_leaves),
                ms(other),
                ms(s.total()),
                ms(b.total() + s.total())
            );
        }
    }
}

/// The leaf-size study: per distribution and refinement target, the tree, the errors
/// and the timings.
fn print_leaf_study(study: &[(Distribution, Vec<Run>)], config: &Config) {
    println!();
    println!("## Leaf-size study");
    println!();
    println!(
        "p = {LEAF_STUDY_DEGREE} in f64, N = {}, `max_level` {}, the problems of the sweeps \
         with only `max_points_per_leaf` changed. Wall time in ms, as in the sweeps; near \
         share = leaves / evaluate (the leaf stage holds P2P, and also L2P and M2P).",
        config.n, config.max_level
    );
    println!();
    println!(
        "| distribution | max points per leaf | levels | leaves | leaf levels (min – max) | points per leaf (min / mean / max) | U pairs | V pairs | W pairs | X pairs | φ L2 | ∇φ L2 | build | upward | downward | leaves | evaluate | near share |"
    );
    println!(
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    for (d, runs) in study {
        for (size, r) in LEAF_SIZES.iter().zip(runs) {
            let s = &r.stages;
            println!(
                "| {} | {size} | {} | {:.3e} | {:.3e} | {} | {} | {} | {} | {} | {:.0}% |",
                d.name(),
                tree_row(r),
                r.potential.l2,
                r.gradient.l2,
                ms(r.build.total()),
                ms(s.upward_local + s.upward_global),
                ms(s.downward),
                ms(s.evaluate_leaves),
                ms(s.total()),
                100.0 * s.evaluate_leaves.as_secs_f64() / s.total().as_secs_f64()
            );
        }
    }
}

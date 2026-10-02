//! Accuracy of the complete FMM on a uniform tree (C3.2) and on adaptive trees of
//! clustered points (C3.3). Prints Markdown on stdout.
//!
//! - **cube** (C3.2): N = 10⁵ points uniform in a cube, a uniform level-4 tree.
//! - **sphere**, **plummer**, **clusters** (C3.3): N = 10⁵ points on a sphere surface,
//!   in a Plummer sphere and in five tight Gaussian clusters, adaptive trees with
//!   `max_level` 16 and 64 points per leaf as the refinement target.
//!
//! Every distribution: sources equal to targets, p ∈ {3, 8, 18} in f64 and p ∈ {3, 8} in
//! f32, against the f64 direct sum at 1,000 sampled targets, over eight charge vectors.
//! Each clustered distribution is compared with the uniform cube at the same p, N and
//! precision, so the cube always runs.
//!
//! Run in release mode, on one rank (the points are not redistributed until C5.1), with
//! `--distribution d` (`cube`, `sphere`, `plummer`, `clusters` or `all`, the default),
//! `--threads n` rayon threads (default 1) and one BLAS thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example fmm_accuracy -- --threads 4
//! ```
//!
//! The problems and the error measure are described in `nd_fmm_validate::fmm_accuracy`.
//! The errors are deterministic for the seed and the same for every number of threads;
//! the timings are wall times on this machine, reported and never asserted. MPI is
//! initialised with `Threading::Funneled`, and the threading report (rayon threads, MPI
//! level, BLAS variables; `nd_fmm_exec::threading`) is printed with the results.

use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::{
    Config, Distribution, Oracle, PREDICTION, Problem, Run, prediction, run,
};

/// The degrees in f64: those of the prediction.
const F64_PS: [usize; 3] = [3, 8, 18];

/// The degrees in f32, at most 8 (design §4).
const F32_PS: [usize; 2] = [3, 8];

/// The command line: the distributions besides the cube, and the threads.
struct Arguments {
    distributions: Vec<Distribution>,
    threads: usize,
}

/// Parses `--distribution d` and `--threads n`, in any order; exits with a message on
/// anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: fmm_accuracy [--distribution cube|sphere|plummer|clusters|all] \
             [--threads n], n >= 1; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        distributions: Distribution::ALL.to_vec(),
        threads: 1,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match (flag.as_str(), value.as_str()) {
            ("--threads", n) => match n.parse() {
                Ok(n) if n >= 1 => parsed.threads = n,
                _ => usage(),
            },
            ("--distribution", "all") => parsed.distributions = Distribution::ALL.to_vec(),
            ("--distribution", name) => match Distribution::from_name(name) {
                Some(d) => parsed.distributions = vec![d],
                None => usage(),
            },
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() {
        usage();
    }
    parsed
}

/// The runs of one distribution: every p in f64, then in f32, and the time of its
/// direct sums.
struct Report {
    config: Config,
    runs: Vec<Run>,
    oracle_time: Duration,
}

/// Draws the problem of `config`, computes its oracles and runs every p.
fn report(config: Config, threads: usize, comm: &SimpleCommunicator) -> Report {
    let start = Instant::now();
    let problem = Problem::new(&config);
    let oracle64 = Oracle::new(&problem, &problem.charges);
    let charges32 = problem.charges_as::<f32>();
    let rounded: Vec<Vec<f64>> = charges32
        .iter()
        .map(|q| q.iter().map(|&v| f64::from(v)).collect())
        .collect();
    let oracle32 = Oracle::new(&problem, &rounded);
    let oracle_time = start.elapsed();
    let mut runs: Vec<Run> = F64_PS
        .iter()
        .map(|&p| {
            run::<f64>(
                &config,
                &problem,
                &problem.charges,
                &oracle64,
                (p, threads),
                comm,
            )
        })
        .collect();
    runs.extend(
        F32_PS
            .iter()
            .map(|&p| run::<f32>(&config, &problem, &charges32, &oracle32, (p, threads), comm)),
    );
    Report {
        config,
        runs,
        oracle_time,
    }
}

/// The uniform-cube run of `cube` at the precision and degree of `r`.
fn reference<'r>(cube: &'r Report, r: &Run) -> &'r Run {
    cube.runs
        .iter()
        .find(|c| c.precision == r.precision && c.p == r.p)
        .expect("the cube runs every precision and p")
}

fn main() {
    let Arguments {
        distributions,
        threads,
    } = arguments();
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if threads > 1 && provided < Threading::Funneled {
        eprintln!("--threads {threads} needs MPI at Funneled; it provides {provided:?}");
        std::process::exit(2);
    }
    if comm.size() != 1 {
        eprintln!("fmm_accuracy runs on one rank; points are not redistributed until C5.1");
        std::process::exit(2);
    }

    let cube = report(Config::C32, threads, &comm);
    let clustered: Vec<Report> = distributions
        .iter()
        .filter(|&&d| d != Distribution::Cube)
        .map(|&d| report(Config::c33(d), threads, &comm))
        .collect();
    let mut reports: Vec<&Report> = Vec::new();
    if distributions.contains(&Distribution::Cube) {
        reports.push(&cube);
    }
    reports.extend(&clustered);

    println!("# Accuracy of the FMM on uniform and adaptive trees (C3.2, C3.3)");
    println!();
    println!(
        "- Problems: N = {} points per distribution, sources equal to targets; {} charge \
         vectors uniform in [-1, 1) (seeds {:#x} + 1 + k for the cube, {:#x} + 1 + k for \
         the others); the domain from `compute_global_bounding_box`; the default M2L \
         strategy; gradients on. The cube has a uniform level-{} tree (`max_level` {}, \
         `max_points_per_leaf` {}); the others adaptive trees (`max_level` {}, \
         `max_points_per_leaf` {}, the same for every distribution).",
        cube.config.n,
        cube.config.charge_vectors,
        cube.config.seed,
        Config::c33(Distribution::Sphere).seed,
        cube.config.max_level,
        cube.config.max_level,
        cube.config.max_points_per_leaf,
        Config::c33(Distribution::Sphere).max_level,
        Config::c33(Distribution::Sphere).max_points_per_leaf,
    );
    println!(
        "- Error measure: relative L2 and max error of φ and of ∇φ (Euclidean norm per \
         target) at {} targets sampled with the seed of the points, against `direct_sum` \
         in f64 over all sources divided by 4π; for each charge vector, then the root \
         mean square over the vectors. \"range\" is the smallest and largest φ L2 error of \
         one vector. f32 runs use the charges rounded to f32, and the oracle on the \
         rounded charges.",
        cube.config.sampled
    );
    println!(
        "- C3.2: the single-translation φ L2 error of P2M → M2L → L2P (design §7; p = 18 \
         re-derived in T9 as the median over 33 source draws, docs/phase3/README.md, \
         \"Predictions\"), {}; the gate allows twice that, on the cube.",
        PREDICTION
            .iter()
            .map(|(p, e)| format!("{e:.2e} at p = {p}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "- C3.3: the φ L2 error of each clustered distribution at most twice that of the \
         uniform cube at the same p and N, in f64; f32 is reported next to it."
    );
    println!(
        "- Machine: {}; {}; {}; one rank, {threads} thread{}. The direct sums took {} s \
         (one thread).",
        cpu_model(),
        cores(),
        target(),
        if threads == 1 { "" } else { "s" },
        std::iter::once(&cube)
            .chain(&clustered)
            .map(|r| format!(
                "{:.1} ({})",
                r.oracle_time.as_secs_f64(),
                r.config.distribution.name()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let threading = &cube.runs[0].threading;
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

    if !clustered.is_empty() {
        print_gate(&cube, &clustered);
    }
    for report in reports {
        print_report(report, &cube);
    }
}

/// The C3.3 summary: per clustered distribution and p in f64, the φ L2 error against
/// the cube's, and the W and X list sizes.
fn print_gate(cube: &Report, clustered: &[Report]) {
    println!();
    println!("## C3.3 gate");
    println!();
    println!(
        "| distribution | precision | p | φ L2 | cube φ L2 | ratio | within 2× | W pairs | X pairs |"
    );
    println!("|---|---|---:|---:|---:|---:|---|---:|---:|");
    for report in clustered {
        for r in &report.runs {
            let c = reference(cube, r);
            let ratio = r.potential.l2 / c.potential.l2;
            let gate = if r.precision == "f64" {
                if ratio <= 2.0 { "yes" } else { "no" }
            } else {
                "(f32, reported)"
            };
            println!(
                "| {} | {} | {} | {:.3e} | {:.3e} | {ratio:.2} | {gate} | {} | {} |",
                r.distribution.name(),
                r.precision,
                r.p,
                r.potential.l2,
                c.potential.l2,
                r.lists.w,
                r.lists.x
            );
        }
    }
}

/// The tree, accuracy and timing tables of one distribution.
fn print_report(report: &Report, cube: &Report) {
    let config = &report.config;
    let tree = &report.runs[0];
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
    println!("### Tree");
    println!();
    let levels: Vec<usize> = (0..tree.leaf_levels.len())
        .filter(|&l| tree.leaf_levels[l] > 0)
        .collect();
    println!(
        "| levels | leaves | leaf levels (min – max) | points per leaf (min / mean / max) | U pairs | V pairs | W pairs | X pairs |"
    );
    println!("|---:|---:|---|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} – {} | {} / {:.1} / {} | {} | {} | {} | {} |",
        tree.nlevels,
        tree.nleaves,
        levels.first().copied().unwrap_or(0),
        levels.last().copied().unwrap_or(0),
        tree.points_per_leaf.0,
        tree.points_per_leaf.1,
        tree.points_per_leaf.2,
        tree.lists.u,
        tree.lists.v,
        tree.lists.w,
        tree.lists.x
    );
    println!();
    println!(
        "Leaves per level: {}.",
        levels
            .iter()
            .map(|&l| format!("{l}: {}", tree.leaf_levels[l]))
            .collect::<Vec<_>>()
            .join(", ")
    );

    println!();
    println!("### Accuracy");
    println!();
    println!(
        "| precision | p | strategy | φ L2 | φ L2 range | φ max | ∇φ L2 | ∇φ max | prediction | φ L2 / prediction | cube φ L2 | φ L2 / cube |"
    );
    println!("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|");
    let is_cube = std::ptr::eq(report, cube);
    for r in &report.runs {
        let (prediction, ratio) = match (prediction(r.p), r.ratio()) {
            (Some(e), Some(ratio)) => (format!("{e:.2e}"), format!("{ratio:.2}")),
            _ => ("—".into(), "—".into()),
        };
        let (cube_l2, cube_ratio) = if is_cube {
            ("—".into(), "—".into())
        } else {
            let c = reference(cube, r);
            (
                format!("{:.3e}", c.potential.l2),
                format!("{:.2}", r.potential.l2 / c.potential.l2),
            )
        };
        println!(
            "| {} | {} | {:?} | {:.3e} | {:.2e} – {:.2e} | {:.3e} | {:.3e} | {:.3e} | {prediction} | {ratio} | {cube_l2} | {cube_ratio} |",
            r.precision,
            r.p,
            r.strategy,
            r.potential.l2,
            r.potential_l2_range.0,
            r.potential_l2_range.1,
            r.potential.max,
            r.gradient.l2,
            r.gradient.max
        );
    }

    println!();
    println!("### Timings");
    println!();
    println!(
        "Wall time in ms. Build: once per row. Evaluate: the mean over the {} charge \
         vectors, by evaluator stage.",
        config.charge_vectors
    );
    println!();
    println!(
        "| precision | p | build | octree | plan | tables | evaluator | load | evaluate | exchange sources | upward local | upward global | exchange multipoles | downward | leaves | output |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let ms = |d: Duration| format!("{:.1}", d.as_secs_f64() * 1e3);
    for r in &report.runs {
        let (b, s) = (&r.build, &r.stages);
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.precision,
            r.p,
            ms(b.total()),
            ms(b.octree),
            ms(b.plan),
            ms(b.tables),
            ms(b.evaluator),
            ms(b.load),
            ms(s.total()),
            ms(s.exchange_sources),
            ms(s.upward_local),
            ms(s.upward_global),
            ms(s.exchange_multipoles),
            ms(s.downward),
            ms(s.evaluate_leaves),
            ms(s.output)
        );
    }
}

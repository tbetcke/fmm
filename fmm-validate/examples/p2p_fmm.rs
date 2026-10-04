//! The SIMD P2P kernel inside the complete FMM, and a leaf-size study with it (Phase 3S
//! T7, C3S.6). Prints Markdown on stdout and progress on stderr.
//!
//! The problems are those of the T12 calibration (`nd_fmm_validate::calibration`): the
//! uniform cube and the Plummer sphere, N = 10⁵ points, sources equal to targets, eight
//! charge vectors, an adaptive tree with `max_level` 16, the default M2L strategy and
//! gradients on, against the f64 direct sum at 1,000 sampled targets.
//!
//! - **Kernels**: 64 points per leaf as the refinement target, p = 3 and 8, f64 and f32,
//!   with each P2P kernel: `reference` (`nd_fmm_ref::p2p`), `auto` (the default, the
//!   widest ISA) and every other available ISA. Per run, the errors, the build and the
//!   stage timings, and the speed-up of the leaf stage and of the evaluation over the
//!   reference.
//! - **Leaf size**: the same problems with `auto` and the refinement targets 16, 32,
//!   64, 128 and 256 (`calibration::LEAF_SIZES`), p = 3 and 8, f64 and f32: the tree,
//!   the errors, the stage timings and the near/far balance; the fastest size per
//!   distribution, degree and precision; and the leaf-size rule of T7
//!   (`calibration::leaf_size_rule`), which reads the one-thread runs.
//!
//! Run in release mode, on one rank, with `--threads n` rayon threads (default 1),
//! `--n N` points (default 10⁵), `--part kernels|leaf|all` (default all) and one BLAS
//! thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example p2p_fmm -- --threads 12
//! ```
//!
//! The errors are deterministic for the seed and the same for every number of threads,
//! and change with the P2P kernel only in the last bits (C3S.5). Timings are wall
//! times on this machine, the mean over the charge vectors, reported and never
//! asserted. MPI is initialised with `Threading::Funneled`.

use std::time::Instant;

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::operator::{Isa, P2pChoice};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::calibration::{
    LEAF_RULE_BASELINE, LEAF_RULE_ERROR_GROWTH, LEAF_RULE_GAIN, LEAF_SIZES,
    LEAF_STUDY_DISTRIBUTIONS, Precision, Reference, config, leaf_size_rule, sweep,
};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Execution, Run};

/// The degrees of every part.
const DEGREES: [usize; 2] = [3, 8];

/// The precisions of every part, f64 first.
const PRECISIONS: [Precision; 2] = [Precision::F64, Precision::F32];

/// Which parts run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Kernels,
    Leaf,
    All,
}

/// The command line.
struct Arguments {
    threads: usize,
    n: usize,
    part: Part,
}

/// Parses `--threads n`, `--n N` and `--part p`, in any order; exits with a message on
/// anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: p2p_fmm [--threads n] [--n N] [--part kernels|leaf|all], n >= 1, \
             N >= 1000; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        threads: 1,
        n: 100_000,
        part: Part::All,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match (flag.as_str(), value.as_str()) {
            ("--threads", n) => match n.parse() {
                Ok(n) if n >= 1 => parsed.threads = n,
                _ => usage(),
            },
            ("--n", n) => match n.parse() {
                Ok(n) if n >= 1000 => parsed.n = n,
                _ => usage(),
            },
            ("--part", "kernels") => parsed.part = Part::Kernels,
            ("--part", "leaf") => parsed.part = Part::Leaf,
            ("--part", "all") => parsed.part = Part::All,
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() {
        usage();
    }
    parsed
}

/// The P2P kernels of the kernel part: the reference, `auto`, then every available
/// ISA other than the one `auto` picks.
fn kernels() -> Vec<P2pChoice> {
    let detected = Isa::detect();
    [P2pChoice::Reference, P2pChoice::Auto]
        .into_iter()
        .chain(
            Isa::available()
                .filter(|&isa| isa != detected)
                .map(P2pChoice::Isa),
        )
        .collect()
}

/// One FMM run of a distribution.
struct Measured {
    distribution: Distribution,
    precision: Precision,
    p: usize,
    max_points_per_leaf: usize,
    choice: P2pChoice,
    run: Run,
}

/// Runs one configuration, with progress on stderr.
fn one(
    config: &Config,
    reference: &Reference,
    (precision, p): (Precision, usize),
    execution: Execution,
    comm: &SimpleCommunicator,
) -> Measured {
    let start = Instant::now();
    let run = sweep(config, reference, precision, &[p], execution, comm).remove(0);
    eprintln!(
        "{} {} p = {p}, {} points per leaf, {}: φ L2 {:.3e}, evaluate {:.1} ms ({:.1} s)",
        config.distribution.name(),
        precision.name(),
        config.max_points_per_leaf,
        execution.p2p,
        run.potential.l2,
        run.stages.total().as_secs_f64() * 1e3,
        start.elapsed().as_secs_f64()
    );
    Measured {
        distribution: config.distribution,
        precision,
        p,
        max_points_per_leaf: config.max_points_per_leaf,
        choice: execution.p2p,
        run,
    }
}

fn main() {
    let Arguments { threads, n, part } = arguments();
    let (universe, provided) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if threads > 1 && provided < Threading::Funneled {
        eprintln!("--threads {threads} needs MPI at Funneled; it provides {provided:?}");
        std::process::exit(2);
    }
    if comm.size() != 1 {
        eprintln!("p2p_fmm runs on one rank; points are not redistributed until C5.1");
        std::process::exit(2);
    }

    let start = Instant::now();
    let mut kernel_runs = Vec::new();
    let mut leaf_runs = Vec::new();
    for &d in &LEAF_STUDY_DISTRIBUTIONS {
        let base = config(d, n);
        let reference = Reference::new(&base);
        eprintln!(
            "{}: oracles in {:.1} s",
            d.name(),
            reference.time.as_secs_f64()
        );
        for precision in PRECISIONS {
            for p in DEGREES {
                if part != Part::Leaf {
                    for choice in kernels() {
                        let execution = Execution {
                            p2p: choice,
                            ..Execution::threads(threads)
                        };
                        kernel_runs.push(one(&base, &reference, (precision, p), execution, &comm));
                    }
                }
                if part != Part::Kernels {
                    for max_points_per_leaf in LEAF_SIZES {
                        let config = Config {
                            max_points_per_leaf,
                            ..base
                        };
                        let execution = Execution::threads(threads);
                        leaf_runs.push(one(&config, &reference, (precision, p), execution, &comm));
                    }
                }
            }
        }
    }
    let wall = start.elapsed();

    let any = kernel_runs
        .first()
        .or(leaf_runs.first())
        .expect("a part ran");
    print_header(&config(Distribution::Cube, n), threads, &any.run, wall);
    if !kernel_runs.is_empty() {
        print_kernels(&kernel_runs);
    }
    if !leaf_runs.is_empty() {
        print_leaf_study(&leaf_runs);
        print_leaf_rule(&leaf_runs, threads);
    }
}

/// Milliseconds with one decimal.
fn ms(seconds: f64) -> String {
    format!("{:.1}", seconds * 1e3)
}

/// The far field (upward and downward), the leaf stage and the evaluation, in seconds.
fn stages(run: &Run) -> (f64, f64, f64, f64) {
    let s = &run.stages;
    (
        (s.upward_local + s.upward_global).as_secs_f64(),
        s.downward.as_secs_f64(),
        s.evaluate_leaves.as_secs_f64(),
        s.total().as_secs_f64(),
    )
}

/// The problems, the machine and the threading.
fn print_header(config: &Config, threads: usize, run: &Run, wall: std::time::Duration) {
    println!("# The SIMD P2P kernel in the FMM, and the leaf size (Phase 3S T7)");
    println!();
    println!(
        "- Problems: the T12 calibration problems, the uniform cube and the Plummer \
         sphere, N = {}, sources equal to targets, {} charge vectors uniform in [-1, 1), \
         an adaptive tree with `max_level` {}, the default M2L strategy (`Auto`), \
         gradients on; p = {DEGREES:?}, f64 and f32.",
        config.n, config.charge_vectors, config.max_level
    );
    println!(
        "- Error measure: relative L2 error of φ and of ∇φ at {} sampled targets against \
         `direct_sum` in f64, the root mean square over the charge vectors (f32 runs: \
         the charges rounded to f32 and their oracle).",
        config.sampled
    );
    println!(
        "- Timings: wall time in ms of one evaluation, the mean over the charge vectors; \
         upward = P2M and M2M, downward = L2L, M2L and P2L, leaves = L2P, M2P and P2P; \
         far = upward + downward, near share = leaves / evaluate."
    );
    println!(
        "- Machine: {}; {}{}; {}; {}; one rank, {threads} thread{}. Wall time of all runs \
         {:.1} s.",
        cpu_model(),
        cores(),
        performance_cores().map_or(String::new(), |p| format!(", {p} performance")),
        target(),
        toolchain(),
        if threads == 1 { "" } else { "s" },
        wall.as_secs_f64()
    );
    println!("- Threading: {}.", run.threading);
    println!(
        "- P2P kernels: `auto` = {} on this machine; ISAs available: {}.",
        P2pChoice::Auto.resolve().expect("auto resolves"),
        Isa::available()
            .map(|isa| isa.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// The kernel part: per distribution, a row per precision, degree and kernel.
fn print_kernels(runs: &[Measured]) {
    println!();
    println!("## P2P kernels in the FMM");
    println!();
    println!(
        "{} points per leaf as the refinement target. \"x\": the reference's time over \
         the kernel's, for the leaf stage and the evaluation. Error ratios: the kernel's \
         error over the reference's.",
        runs[0].max_points_per_leaf
    );
    println!();
    println!(
        "| distribution | precision | p | P2P | ran as | φ L2 | ∇φ L2 | φ / ref | ∇φ / ref | build | upward | downward | leaves | evaluate | near share | leaves x | evaluate x |"
    );
    println!(
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for m in runs {
        let reference = runs
            .iter()
            .find(|r| {
                r.distribution == m.distribution
                    && r.precision == m.precision
                    && r.p == m.p
                    && r.choice == P2pChoice::Reference
            })
            .expect("the reference runs in every configuration");
        let (up, down, leaves, total) = stages(&m.run);
        let (_, _, ref_leaves, ref_total) = stages(&reference.run);
        println!(
            "| {} | {} | {} | {} | {} | {:.3e} | {:.3e} | {:.6} | {:.6} | {} | {} | {} | {} | {} | {:.0}% | {:.2} | {:.2} |",
            m.distribution.name(),
            m.precision.name(),
            m.p,
            m.choice,
            m.run.p2p,
            m.run.potential.l2,
            m.run.gradient.l2,
            m.run.potential.l2 / reference.run.potential.l2,
            m.run.gradient.l2 / reference.run.gradient.l2,
            ms(m.run.build.total().as_secs_f64()),
            ms(up),
            ms(down),
            ms(leaves),
            ms(total),
            100.0 * leaves / total,
            ref_leaves / leaves,
            ref_total / total,
        );
    }
}

/// The leaf-size study: every run, then the fastest size per configuration.
fn print_leaf_study(runs: &[Measured]) {
    println!();
    println!("## Leaf-size study (P2P `auto`)");
    println!();
    println!(
        "| distribution | precision | p | max points per leaf | leaves | points per leaf (min / mean / max) | V pairs | φ L2 | ∇φ L2 | upward | downward | far | leaves | evaluate | near share |"
    );
    println!(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for m in runs {
        let (up, down, leaves, total) = stages(&m.run);
        let r = &m.run;
        println!(
            "| {} | {} | {} | {} | {} | {} / {:.1} / {} | {} | {:.3e} | {:.3e} | {} | {} | {} | {} | {} | {:.0}% |",
            m.distribution.name(),
            m.precision.name(),
            m.p,
            m.max_points_per_leaf,
            r.nleaves,
            r.points_per_leaf.0,
            r.points_per_leaf.1,
            r.points_per_leaf.2,
            r.lists.v,
            r.potential.l2,
            r.gradient.l2,
            ms(up),
            ms(down),
            ms(up + down),
            ms(leaves),
            ms(total),
            100.0 * leaves / total,
        );
    }
    println!();
    println!("### Fastest leaf size per configuration");
    println!();
    println!(
        "| distribution | precision | p | fastest | evaluate (ms) | at {LEAF_RULE_BASELINE} (ms) | {LEAF_RULE_BASELINE} / fastest | evaluate at {} (ms) |",
        LEAF_SIZES
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |");
    for (d, precision, p) in configurations() {
        let mine = configuration(runs, d, precision, p);
        let time = |m: &Measured| stages(&m.run).3;
        let fastest = mine
            .iter()
            .min_by(|a, b| time(a).total_cmp(&time(b)))
            .expect("every size runs");
        let base = mine
            .iter()
            .find(|m| m.max_points_per_leaf == LEAF_RULE_BASELINE)
            .expect("the baseline size runs");
        println!(
            "| {} | {} | {p} | {} | {} | {} | {:.2} | {} |",
            d.name(),
            precision.name(),
            fastest.max_points_per_leaf,
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

/// The configurations of the rule, in a fixed order.
fn configurations() -> Vec<(Distribution, Precision, usize)> {
    LEAF_STUDY_DISTRIBUTIONS
        .iter()
        .flat_map(|&d| {
            PRECISIONS
                .iter()
                .flat_map(move |&precision| DEGREES.iter().map(move |&p| (d, precision, p)))
        })
        .collect()
}

/// The runs of one configuration, in the order of [`LEAF_SIZES`].
fn configuration(
    runs: &[Measured],
    d: Distribution,
    precision: Precision,
    p: usize,
) -> Vec<&Measured> {
    LEAF_SIZES
        .iter()
        .map(|&size| {
            runs.iter()
                .find(|m| {
                    m.distribution == d
                        && m.precision == precision
                        && m.p == p
                        && m.max_points_per_leaf == size
                })
                .expect("every size runs in every configuration")
        })
        .collect()
}

/// The leaf-size rule over the study.
fn print_leaf_rule(runs: &[Measured], threads: usize) {
    let configurations = configurations();
    let per_size: Vec<Vec<&Measured>> = (0..LEAF_SIZES.len())
        .map(|i| {
            configurations
                .iter()
                .map(|&(d, precision, p)| configuration(runs, d, precision, p)[i])
                .collect()
        })
        .collect();
    let times: Vec<Vec<f64>> = per_size
        .iter()
        .map(|v| v.iter().map(|m| stages(&m.run).3).collect())
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
    println!("### The leaf-size rule of T7");
    println!();
    println!(
        "Geometric mean of the evaluation time over the {} configurations (cube and \
         Plummer, p = 3 and 8, f64 and f32){}. The fastest size replaces \
         {LEAF_RULE_BASELINE} only if it is at least {:.0}% faster by this measure and no \
         φ or ∇φ L2 error at it is more than {:.0}% worse than at {LEAF_RULE_BASELINE}.",
        configurations.len(),
        if threads == 1 {
            ", one thread: the measure of the rule"
        } else {
            "; the rule reads one-thread runs, so this run only informs it"
        },
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
         {:.3}x. **Chosen: {}**{}.",
        choice.fastest,
        choice.gain,
        choice.error_growth,
        choice.chosen,
        if threads == 1 {
            ""
        } else {
            " (informative; not the measure of the rule)"
        }
    );
}

//! Accuracy of the complete FMM on a uniform tree (C3.2): N = 10⁵ points uniform in a
//! cube, sources equal to targets, a uniform level-4 tree, p ∈ {3, 8, 18} in f64 and
//! p ∈ {3, 8} in f32, against the f64 direct sum at 1,000 sampled targets, over eight
//! charge vectors. Prints Markdown on stdout.
//!
//! Run in release mode, on one rank (the points are not redistributed until C5.1), with
//! `--threads n` rayon threads (default 1) and one BLAS thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example fmm_accuracy -- --threads 4
//! ```
//!
//! The problem and the error measure are described in `nd_fmm_validate::fmm_accuracy`.
//! The errors are deterministic for the seed and the same for every number of threads;
//! the timings are wall times on this machine, reported and never asserted. MPI is
//! initialised with `Threading::Funneled`, and the threading report (rayon threads, MPI
//! level, BLAS variables; `nd_fmm_exec::threading`) is printed with the results.

use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::{Config, Oracle, PREDICTION, Problem, Run, run};

/// The degrees in f64: those of the prediction.
const F64_PS: [usize; 3] = [3, 8, 18];

/// The degrees in f32, at most 8 (design §4).
const F32_PS: [usize; 2] = [3, 8];

/// The `--threads n` argument, 1 by default; exits with a message on anything else.
fn threads() -> usize {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!("usage: fmm_accuracy [--threads n], n >= 1; got {args:?}");
        std::process::exit(2);
    };
    match args.as_slice() {
        [] => 1,
        [flag, n] if flag == "--threads" => match n.parse() {
            Ok(n) if n >= 1 => n,
            _ => usage(),
        },
        _ => usage(),
    }
}

fn main() {
    let threads = threads();
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
    let config = Config::C32;
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
                &comm,
            )
        })
        .collect();
    runs.extend(F32_PS.iter().map(|&p| {
        run::<f32>(
            &config,
            &problem,
            &charges32,
            &oracle32,
            (p, threads),
            &comm,
        )
    }));

    println!("# Accuracy of the FMM on a uniform tree (C3.2)");
    println!();
    println!(
        "- Problem: N = {} points uniform in [-1, 1)^3, sources equal to targets; {} charge \
         vectors uniform in [-1, 1) (seeds {:#x} + 1 + k); a uniform level-{} tree \
         (`max_level` {}, `max_points_per_leaf` 1); the domain from \
         `compute_global_bounding_box`; the default M2L strategy; gradients on.",
        config.n, config.charge_vectors, config.seed, config.max_level, config.max_level
    );
    println!(
        "- Error measure: relative L2 and max error of φ and of ∇φ (Euclidean norm per \
         target) at {} targets sampled with seed {:#x}, against `direct_sum` in f64 over \
         all sources divided by 4π; for each charge vector, then the root mean square over \
         the vectors. \"range\" is the smallest and largest φ L2 error of one vector. f32 \
         runs use the charges rounded to f32, and the oracle on the rounded charges.",
        config.sampled, config.seed
    );
    println!(
        "- Prediction: the single-translation φ L2 error of P2M → M2L → L2P (design §7; \
         p = 18 re-derived in T9 as the median over 33 source draws, docs/phase3/README.md, \
         \"Predictions\"), {}; the C3.2 gate allows twice that.",
        PREDICTION
            .iter()
            .map(|(p, e)| format!("{e:.2e} at p = {p}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "- Machine: {}; {}; {}; one rank, {threads} thread{}. The direct sums took {:.1} \
         s (one thread).",
        cpu_model(),
        cores(),
        target(),
        if threads == 1 { "" } else { "s" },
        oracle_time.as_secs_f64()
    );
    let threading = &runs[0].threading;
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

    let tree = &runs[0];
    println!();
    println!("## Tree");
    println!();
    println!(
        "| levels | leaves | points per leaf (min / mean / max) | U pairs | V pairs | W pairs | X pairs |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} / {:.1} / {} | {} | {} | {} | {} |",
        tree.nlevels,
        tree.nleaves,
        tree.points_per_leaf.0,
        tree.points_per_leaf.1,
        tree.points_per_leaf.2,
        tree.lists.u,
        tree.lists.v,
        tree.lists.w,
        tree.lists.x
    );

    println!();
    println!("## Accuracy");
    println!();
    println!(
        "| precision | p | strategy | φ L2 | φ L2 range | φ max | ∇φ L2 | ∇φ max | prediction | φ L2 / prediction | within 2× |"
    );
    println!("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|");
    for r in &runs {
        let (prediction, ratio, gate) =
            match (nd_fmm_validate::fmm_accuracy::prediction(r.p), r.ratio()) {
                (Some(e), Some(ratio)) => (
                    format!("{e:.2e}"),
                    format!("{ratio:.2}"),
                    if ratio <= 2.0 { "yes" } else { "no" }.to_string(),
                ),
                _ => ("—".into(), "—".into(), "—".into()),
            };
        println!(
            "| {} | {} | {:?} | {:.3e} | {:.2e} – {:.2e} | {:.3e} | {:.3e} | {:.3e} | {prediction} | {ratio} | {gate} |",
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
    println!("## Timings");
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
    for r in &runs {
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

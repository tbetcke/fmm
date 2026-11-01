//! The Helmholtz FMM against the sampled direct sum on the reference workloads
//! (Helmholtz CH3.1c; docs/design/helmholtz-fmm-proposal.md §9.1 and §10, CH3.1). Prints
//! Markdown on stdout.
//!
//! One case per kD and ε: the workload's points as sources and targets, the degrees of
//! S7's closed form at ε (or named, the same on every level ≥ 2), real or complex charges,
//! m vectors, gradients, against `nd_fmm_ref::helmholtz::p2p::direct_sum` in f64 at the
//! sampled targets (1,000 by default). Each case runs at every thread count of
//! `--threads`, and each row prints the tree, the degrees, the build and evaluation times
//! (slowest rank), the relative ℓ2 errors of φ and ∇φ over all vectors, and the
//! rank-invariant checksum of the whole output: equal checksums across thread counts and
//! rank counts show the run deterministic (`nd_fmm_validate::helmholtz_accuracy`).
//!
//! Options (each value may be a comma-separated list where marked):
//!
//! - `--workload w`: `w1` (default), `w2`, `w3` (ten points per wavelength), `w3:<n>`,
//!   `w4-<cube|sphere|plummer|clusters>:<n>`;
//! - `--kd x,…`: Re kD (default 10); `--kim y`: Im kD (default 0);
//! - `--eps e,…`: the tolerances (default 1e-3); `--degree p` names p on every level ≥ 2
//!   in place of a tolerance;
//! - `--charges real|complex` (default complex), `--vectors m` (default 1),
//!   `--samples n` (default 1000), `--no-gradients`;
//! - `--threads t,…` (default 1), `--oracle-threads t` (default the largest of
//!   `--threads`).
//!
//! Run in release mode, on any number of ranks (each rank passes a contiguous block of
//! the points, the oracle is sharded over the ranks, rank 0 prints), with one BLAS
//! thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_accuracy -- \
//!     --workload w1 --kd 30,100 --eps 1e-3,1e-6 --threads 1,6
//! mpirun -n 3 target/release/examples/helmholtz_accuracy --workload w1 --kd 100 --threads 2
//! ```
//!
//! MPI is initialised with `Threading::Funneled`. Timings are wall times on this machine,
//! reported and never asserted.

use std::time::Instant;

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::helmholtz::{ChargeKind, Degrees};
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{Case, Oracle, Problem, complex_k, run};
use nd_fmm_validate::helmholtz_workloads::Workload;

/// The command line.
struct Arguments {
    workload: String,
    kds: Vec<f64>,
    kim: f64,
    degrees: Vec<Degrees>,
    charges: ChargeKind,
    vectors: usize,
    samples: usize,
    gradients: bool,
    threads: Vec<usize>,
    oracle_threads: Option<usize>,
}

/// Parses a comma-separated list.
fn list<T: std::str::FromStr>(value: &str) -> Option<Vec<T>> {
    value.split(',').map(|v| v.parse().ok()).collect()
}

/// The workload of a name at kD (W1 and W3 resolved depend on it).
fn workload(name: &str, kd: f64) -> Option<Workload> {
    let count = |n: &str| n.parse::<usize>().ok().filter(|&n| n > 0);
    match name.split_once(':') {
        None => match name {
            "w1" => Some(Workload::W1),
            "w2" => Some(Workload::W2),
            "w3" => Some(Workload::w3_resolved(kd)),
            _ => None,
        },
        Some(("w3", n)) => Some(Workload::W3 { n: count(n)? }),
        Some((w4, n)) => Some(Workload::W4 {
            distribution: Distribution::from_name(w4.strip_prefix("w4-")?)?,
            n: count(n)?,
        }),
    }
}

/// Parses the options of the module documentation; exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_accuracy [--workload w1|w2|w3|w3:<n>|w4-<distribution>:<n>] \
             [--kd x,…] [--kim y] [--eps e,…|--degree p] [--charges real|complex] \
             [--vectors m] [--samples n] [--no-gradients] [--threads t,…] \
             [--oracle-threads t]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        workload: "w1".into(),
        kds: vec![10.0],
        kim: 0.0,
        degrees: vec![Degrees::Tolerance(1e-3)],
        charges: ChargeKind::Complex,
        vectors: 1,
        samples: 1000,
        gradients: true,
        threads: vec![1],
        oracle_threads: None,
    };
    let mut rest = args.as_slice();
    loop {
        rest = match rest {
            [] => break,
            [flag, tail @ ..] if flag == "--no-gradients" => {
                parsed.gradients = false;
                tail
            }
            [flag, value, tail @ ..] => {
                let ok = match flag.as_str() {
                    "--workload" => {
                        parsed.workload = value.clone();
                        workload(value, 1.0).is_some()
                    }
                    "--kd" => list(value).map(|v| parsed.kds = v).is_some(),
                    "--kim" => value.parse().map(|v| parsed.kim = v).is_ok(),
                    "--eps" => list::<f64>(value)
                        .map(|v| parsed.degrees = v.into_iter().map(Degrees::Tolerance).collect())
                        .is_some(),
                    "--degree" => value
                        .parse::<usize>()
                        .map(|p| parsed.degrees = vec![Degrees::PerLevel(vec![p; 64])])
                        .is_ok(),
                    "--charges" => match value.as_str() {
                        "real" => Some(ChargeKind::Real),
                        "complex" => Some(ChargeKind::Complex),
                        _ => None,
                    }
                    .map(|c| parsed.charges = c)
                    .is_some(),
                    "--vectors" => value.parse().map(|v| parsed.vectors = v).is_ok(),
                    "--samples" => value.parse().map(|v| parsed.samples = v).is_ok(),
                    "--threads" => list(value).map(|v| parsed.threads = v).is_some(),
                    "--oracle-threads" => value
                        .parse()
                        .map(|v| parsed.oracle_threads = Some(v))
                        .is_ok(),
                    _ => false,
                };
                if !ok {
                    usage();
                }
                tail
            }
            _ => usage(),
        };
    }
    if parsed.vectors == 0 || parsed.threads.contains(&0) || parsed.kds.is_empty() {
        usage();
    }
    parsed
}

/// The degrees as a short string: runs of equal degrees as `p×count`.
fn degrees_string(degrees: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &p in degrees {
        match runs.last_mut() {
            Some((q, count)) if *q == p => *count += 1,
            _ => runs.push((p, 1)),
        }
    }
    runs.iter()
        .map(|&(p, count)| {
            if count == 1 {
                p.to_string()
            } else {
                format!("{p}×{count}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let arguments = arguments();
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    let root = world.rank() == 0;
    let oracle_threads = arguments
        .oracle_threads
        .unwrap_or_else(|| *arguments.threads.iter().max().unwrap_or(&1));
    if root {
        println!("# Helmholtz FMM against the sampled direct sum");
        println!();
        println!(
            "Machine: {}, {}, {}; {} ranks; BLAS threads: OPENBLAS_NUM_THREADS={}",
            cpu_model(),
            cores(),
            target(),
            world.size(),
            std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        );
        println!();
        println!(
            "| Workload | N | kD | degrees | charges | m | ranks × threads | levels | leaf level \
             | p by level | leaves | max leaf | build s | evaluate s | φ | ∇φ | checksum |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: \
             | ---: | ---: | ---: | --- |"
        );
    }
    for &kd in &arguments.kds {
        let workload = workload(&arguments.workload, kd).expect("checked");
        for degrees in &arguments.degrees {
            let mut case = Case::new(workload, (kd, arguments.kim), degrees.clone());
            case.charges = arguments.charges;
            case.vectors = arguments.vectors;
            case.samples = arguments.samples;
            case.gradients = arguments.gradients;
            let problem = Problem::new(&case);
            let start = Instant::now();
            let oracle = Oracle::sharded(
                complex_k(&case),
                &problem,
                case.gradients,
                oracle_threads,
                &world,
            );
            let oracle_seconds = start.elapsed().as_secs_f64();
            let kd_string = if arguments.kim == 0.0 {
                format!("{kd}")
            } else {
                format!("{kd} + {}i", arguments.kim)
            };
            let degrees_name = match degrees {
                Degrees::Tolerance(eps) => format!("ε = {eps:e}"),
                Degrees::PerLevel(p) => format!("p = {}", p[0]),
            };
            let charges = match case.charges {
                ChargeKind::Real => "real",
                ChargeKind::Complex => "complex",
            };
            for &threads in &arguments.threads {
                match run(&case, &problem, &oracle, threads, &world) {
                    Ok(row) if root => {
                        let all = row.errors.last().expect("an entry over all vectors");
                        let gradient = if case.gradients {
                            format!("{:.2e}", all[1])
                        } else {
                            "–".into()
                        };
                        println!(
                            "| {} | {} | {kd_string} | {degrees_name} | {charges} | {} | {} × {} \
                             | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2e} | {gradient} \
                             | {:016x} |",
                            workload.name(),
                            problem.len(),
                            case.vectors,
                            row.ranks,
                            row.threads,
                            row.levels,
                            row.min_leaf_level,
                            degrees_string(&row.degrees),
                            row.leaves,
                            row.max_leaf_points,
                            row.build,
                            row.evaluate,
                            all[0],
                            row.checksum,
                        );
                        if case.vectors > 1 {
                            let per: Vec<String> = row.errors[..case.vectors]
                                .iter()
                                .map(|e| format!("{:.2e}/{:.2e}", e[0], e[1]))
                                .collect();
                            println!("|  | per vector φ/∇φ: {} |", per.join(", "));
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        if root {
                            eprintln!("{} at kD = {kd_string}: {error}", workload.name());
                        }
                    }
                }
            }
            if root {
                eprintln!(
                    "{} kD = {kd_string} {degrees_name}: oracle {oracle_seconds:.1} s",
                    workload.name()
                );
            }
        }
    }
}

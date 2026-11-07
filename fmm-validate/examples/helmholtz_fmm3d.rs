//! The Helmholtz FMM beside FMM3D's `hfmm3d` on the reference workloads (Helmholtz CH3.6;
//! docs/design/helmholtz-fmm-proposal.md §9.2, the H3 row). Prints Markdown on stdout.
//!
//! Three steps around `tools/fmm3d/driver.f90`'s `helmholtz` mode (`tools/fmm3d/run.sh
//! helmholtz` runs the first two and the driver; `tools/helmholtz/ch6-2d-runs.sh` the
//! first and the third):
//!
//! ```text
//! helmholtz_fmm3d problem CASE --out FILE
//! fmm3d_driver helmholtz FILE PREFIX K 0 1e-3,1e-6,1e-12 1 1      # FMM3D, FULL = 1
//! helmholtz_fmm3d compare CASE --eps 1e-3,1e-6 --fmm3d default=PREFIX \
//!     --reference 1e-12 [--problem FILE] [--threads t]
//! helmholtz_fmm3d errors CASE --eps 1e-3,1e-6 --fmm3d ps=PREFIX1,pw3=PREFIX2,… \
//!     [--problem FILE] [--threads t]
//! ```
//!
//! `problem` writes the case's points, its one charge vector and its sampled targets as
//! the driver's problem file. `compare` runs our FMM on the same case at each ε
//! (`Degrees::Tolerance`, the default rule) and measures it and FMM3D's output at the same
//! ε (`PREFIX-<ε>.out`, its seconds from the driver's log `PREFIX.log`) against the same
//! direct sum at the same sampled targets (`helmholtz_accuracy::Oracle`); with
//! `--reference e`, FMM3D at a tight eps from the first `--fmm3d` (`PREFIX-<e>.full.out`,
//! its errors at the samples from `PREFIX-<e>.out` printed), it also compares our output
//! and FMM3D's full outputs (`PREFIX-<ε>.full.out`) with it at every point: the relative
//! ℓ2 errors over all targets and the worst box of the FMM's deepest level, over ε and
//! over the ℓ2 error (how far one box stands out)
//! (`nd_fmm_validate::helmholtz_fmm3d`). `errors` (Helmholtz CH6.2d) measures FMM3D's
//! outputs alone, without our FMM: per token of `--eps` and run of `--fmm3d`, its errors at
//! the samples against the same direct sum, its seconds, and its errors over the first
//! run's at the same token (CH6.2d's scan: the first run FMM3D with every level on its
//! point-and-shoot M2L, the others one level each on plane waves). A full output's values
//! at the sampled targets must equal the sampled output's bit for bit, and with
//! `--problem` the file must be the case's byte for byte; anything else stops the run on
//! every rank before MPI's first collective.
//!
//! CASE is:
//!
//! - `--workload w`: `w1` (default), `w2`, `w3` (ten points per wavelength), `w3:<n>`,
//!   `w4-<cube|sphere|plummer|clusters>:<n>`;
//! - `--kd x`: Re kD (default 10); `--kim y`: Im kD (default 0);
//! - `--charges real|complex` (default complex), `--samples n` (default 1000).
//!
//! `compare` also takes `--eps e,…` (default 1e-3; each value is also the token of
//! FMM3D's files, as given), `--fmm3d label=PREFIX,…` (FMM3D's runs, for example one per
//! build variant; none to run ours alone), `--threads t` (default 1),
//! `--oracle-threads t` (default `--threads`) and `--p2p` (default `auto`).
//!
//! Run in release mode, on any number of ranks for `compare` (each rank passes a
//! contiguous block of the points, rank 0 prints), with one BLAS thread. One charge
//! vector, gradients on. Timings are wall times on this machine: ours the build and the
//! evaluation (slowest rank), FMM3D's the whole call (its setup included; it keeps
//! nothing between calls). Reported, never asserted.

use std::path::{Path, PathBuf};

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::helmholtz::{ChargeKind, Degrees};
use nd_fmm_exec::operator::P2pChoice;
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{Case, Oracle, Problem, complex_k, run_output};
use nd_fmm_validate::helmholtz_fmm3d::{
    Crosscheck, Grid, Values, crosscheck, crosscheck_values, problem_bytes, read_full,
    read_sampled, sample_errors, seconds,
};
use nd_fmm_validate::helmholtz_workloads::{SEED, Workload};

/// The command line.
struct Arguments {
    step: String,
    workload: String,
    kd: f64,
    kim: f64,
    charges: ChargeKind,
    samples: usize,
    out: Option<PathBuf>,
    problem: Option<PathBuf>,
    eps: Vec<String>,
    fmm3d: Vec<(String, PathBuf)>,
    reference: Option<String>,
    threads: usize,
    oracle_threads: Option<usize>,
    p2p: P2pChoice,
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
            "usage: helmholtz_fmm3d problem|compare|errors [--workload w1|w2|w3|w3:<n>|w4-<distribution>:<n>] \
             [--kd x] [--kim y] [--charges real|complex] [--samples n] [--out FILE] \
             [--eps e,…] [--fmm3d label=PREFIX,…] [--reference e] [--problem FILE] \
             [--threads t] [--oracle-threads t] [--p2p auto|reference|<isa>]; got {args:?}"
        );
        std::process::exit(2);
    };
    let Some((step, mut rest)) = args.split_first() else {
        usage()
    };
    let mut parsed = Arguments {
        step: step.clone(),
        workload: "w1".into(),
        kd: 10.0,
        kim: 0.0,
        charges: ChargeKind::Complex,
        samples: 1000,
        out: None,
        problem: None,
        eps: vec!["1e-3".into()],
        fmm3d: Vec::new(),
        reference: None,
        threads: 1,
        oracle_threads: None,
        p2p: P2pChoice::Auto,
    };
    while let [flag, value, tail @ ..] = rest {
        let ok = match flag.as_str() {
            "--workload" => {
                parsed.workload = value.clone();
                workload(value, 1.0).is_some()
            }
            "--kd" => value.parse().map(|v| parsed.kd = v).is_ok(),
            "--kim" => value.parse().map(|v| parsed.kim = v).is_ok(),
            "--charges" => match value.as_str() {
                "real" => Some(ChargeKind::Real),
                "complex" => Some(ChargeKind::Complex),
                _ => None,
            }
            .map(|c| parsed.charges = c)
            .is_some(),
            "--samples" => value.parse().map(|v| parsed.samples = v).is_ok(),
            "--out" => {
                parsed.out = Some(value.into());
                true
            }
            "--problem" => {
                parsed.problem = Some(value.into());
                true
            }
            "--eps" => {
                parsed.eps = value.split(',').map(str::to_owned).collect();
                parsed
                    .eps
                    .iter()
                    .all(|e| e.parse::<f64>().is_ok_and(|e| e > 0.0))
            }
            "--fmm3d" => value
                .split(',')
                .map(|pair| pair.split_once('=').map(|(l, p)| (l.to_owned(), p.into())))
                .collect::<Option<Vec<_>>>()
                .map(|v| parsed.fmm3d = v)
                .is_some(),
            "--reference" => {
                parsed.reference = Some(value.clone());
                value.parse::<f64>().is_ok_and(|e| e > 0.0)
            }
            "--threads" => value.parse().map(|v| parsed.threads = v).is_ok(),
            "--oracle-threads" => value
                .parse()
                .map(|v| parsed.oracle_threads = Some(v))
                .is_ok(),
            "--p2p" => value
                .parse::<P2pChoice>()
                .ok()
                .filter(|c| c.resolve().is_ok())
                .map(|c| parsed.p2p = c)
                .is_some(),
            _ => false,
        };
        if !ok {
            usage();
        }
        rest = tail;
    }
    let valid = rest.is_empty()
        && parsed.threads > 0
        && parsed.samples > 0
        && match parsed.step.as_str() {
            "problem" => parsed.out.is_some(),
            "compare" => parsed.reference.is_none() || !parsed.fmm3d.is_empty(),
            "errors" => parsed.reference.is_none() && !parsed.fmm3d.is_empty(),
            _ => false,
        };
    if !valid {
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

/// Exits with status 1 after `message`.
fn fail(message: String) -> ! {
    eprintln!("helmholtz_fmm3d: {message}");
    std::process::exit(1);
}

/// One FMM3D output: its sampled values, its full values if written, its seconds and
/// threads.
struct Fmm3dRun {
    sampled: Values,
    full: Option<Values>,
    seconds: Option<(f64, usize)>,
}

/// Reads `PREFIX-<token>.out`, `PREFIX-<token>.full.out` (if present) and the seconds of
/// `token` in `PREFIX.log` (if present); checks the full output at the samples against
/// the sampled output, bit for bit.
fn read_run(prefix: &Path, token: &str, problem: &Problem) -> Fmm3dRun {
    let name = |suffix: &str| PathBuf::from(format!("{}{suffix}", prefix.display()));
    let sampled =
        read_sampled(&name(&format!("-{token}.out"))).unwrap_or_else(|e| fail(e.to_string()));
    if sampled.len() != problem.sample.len() {
        fail(format!(
            "{}: {} samples, the case has {}",
            prefix.display(),
            sampled.len(),
            problem.sample.len()
        ));
    }
    let full_path = name(&format!("-{token}.full.out"));
    let full = full_path.exists().then(|| {
        let full = read_full(&full_path).unwrap_or_else(|e| fail(e.to_string()));
        if full.len() != problem.len() {
            fail(format!(
                "{}: {} points, the case has {}",
                full_path.display(),
                full.len(),
                problem.len()
            ));
        }
        let at = full.at(&problem.sample);
        let same = |a: &[nd_fmm_math::Complex<f64>], b: &[nd_fmm_math::Complex<f64>]| {
            a.iter()
                .zip(b)
                .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits())
        };
        let gradients_same = at
            .gradient
            .iter()
            .zip(&sampled.gradient)
            .all(|(a, b)| same(a, b));
        if !same(&at.potential, &sampled.potential) || !gradients_same {
            fail(format!(
                "{}: the full output differs from the sampled one",
                full_path.display()
            ));
        }
        full
    });
    let seconds = std::fs::read_to_string(name(".log"))
        .ok()
        .and_then(|log| seconds(&log, token));
    Fmm3dRun {
        sampled,
        full,
        seconds,
    }
}

/// A crosscheck's cells: the ℓ2 errors over all targets, the worst box (φ, ∇φ) over ε,
/// and over the ℓ2 errors with the number of boxes; dashes without one.
fn crosscheck_cells(check: Option<&Crosscheck>, eps: f64) -> String {
    match check {
        Some(c) => {
            let worst = c.worst();
            let over = |j: usize| {
                if c.errors[j] > 0.0 {
                    format!("{:.1}", worst[j] / c.errors[j])
                } else {
                    "–".into()
                }
            };
            format!(
                "{:.2e} | {:.2e} | {:.2} / {:.2} | {} / {} of {}",
                c.errors[0],
                c.errors[1],
                worst[0] / eps,
                worst[1] / eps,
                over(0),
                over(1),
                c.boxes.len()
            )
        }
        None => "– | – | – | –".into(),
    }
}

/// The `count` worst boxes of `check` in field `j` (0 φ, 1 ∇φ) as `(i,j,k) e/ℓ2`, and how
/// many boxes exceed 10× the ℓ2 error, for the log.
fn worst_boxes(check: &Crosscheck, j: usize, count: usize) -> String {
    let mut boxes = check.boxes.clone();
    boxes.sort_by(|a, b| b.1[j].total_cmp(&a.1[j]));
    let l2 = check.errors[j];
    let cell = |key: u64, a: usize| (key >> (21 * a)) & ((1 << 21) - 1);
    let worst: Vec<String> = boxes
        .iter()
        .take(count)
        .map(|&(k, e)| {
            format!(
                "({},{},{}) {:.1}",
                cell(k, 0),
                cell(k, 1),
                cell(k, 2),
                e[j] / l2
            )
        })
        .collect();
    let above = boxes.iter().filter(|(_, e)| e[j] > 10.0 * l2).count();
    format!(
        "{}; {above} of {} boxes above 10× the ℓ2 error",
        worst.join(", "),
        boxes.len()
    )
}

/// The step `errors`: FMM3D's runs at the samples against the case's oracle, one row per
/// token and run, with each run's errors over the first run's at the token.
fn errors(arguments: &Arguments, case: &Case, problem: &Problem, workload: Workload) {
    let runs: Vec<Vec<Fmm3dRun>> = arguments
        .fmm3d
        .iter()
        .map(|(_, prefix)| {
            arguments
                .eps
                .iter()
                .map(|e| read_run(prefix, e, problem))
                .collect()
        })
        .collect();
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    let oracle = Oracle::sharded(
        complex_k(case),
        problem,
        true,
        arguments.oracle_threads.unwrap_or(arguments.threads),
        &world,
    );
    if world.rank() != 0 {
        return;
    }
    println!(
        "# FMM3D's errors: {} at kD = {} + {}i",
        workload.name(),
        arguments.kd,
        arguments.kim
    );
    println!();
    println!(
        "N = {}, {} sampled targets; the ratios are each run's errors over the first run's \
         ({}) at the same eps",
        problem.len(),
        problem.sample.len(),
        arguments.fmm3d[0].0
    );
    println!();
    println!("| Workload | N | kD | eps | run | threads | s | φ | ∇φ | φ ÷ first | ∇φ ÷ first |");
    println!("| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (j, token) in arguments.eps.iter().enumerate() {
        let first = sample_errors(&runs[0][j].sampled, &oracle, 0);
        for ((label, _), run) in arguments.fmm3d.iter().zip(&runs) {
            let run = &run[j];
            let e = sample_errors(&run.sampled, &oracle, 0);
            let (time, threads) = run.seconds.map_or(("–".into(), "–".into()), |(s, t)| {
                (format!("{s:.3}"), t.to_string())
            });
            println!(
                "| {} | {} | {} | {token} | {label} | {threads} | {time} | {:.3e} | {:.3e} | {:.3} | {:.3} |",
                workload.name(),
                problem.len(),
                arguments.kd,
                e[0],
                e[1],
                e[0] / first[0],
                e[1] / first[1],
            );
        }
    }
}

fn main() {
    let arguments = arguments();
    let workload = workload(&arguments.workload, arguments.kd).expect("checked");
    let mut case = Case::new(
        workload,
        (arguments.kd, arguments.kim),
        Degrees::Tolerance(1e-3),
    );
    case.charges = arguments.charges;
    case.samples = arguments.samples;
    case.p2p = arguments.p2p;
    let problem = Problem::new(&case);
    let bytes = problem_bytes(&problem, 0, SEED);
    if arguments.step == "problem" {
        let out = arguments.out.as_ref().expect("checked");
        std::fs::write(out, &bytes).unwrap_or_else(|e| fail(format!("{}: {e}", out.display())));
        println!(
            "helmholtz_fmm3d: {} at kD = {} + {}i: {} points, {} samples -> {}",
            workload.name(),
            arguments.kd,
            arguments.kim,
            problem.len(),
            problem.sample.len(),
            out.display()
        );
        return;
    }

    // Every file is read on every rank before MPI starts, so that a missing or
    // mismatched one stops every rank alike.
    if let Some(path) = &arguments.problem {
        let file = std::fs::read(path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
        if file != bytes {
            fail(format!(
                "{} is not this case's problem file",
                path.display()
            ));
        }
    }
    if arguments.step == "errors" {
        errors(&arguments, &case, &problem, workload);
        return;
    }
    let reference = arguments.reference.as_ref().map(|token| {
        let prefix = &arguments.fmm3d[0].1;
        let run = read_run(prefix, token, &problem);
        if run.full.is_none() {
            fail(format!("{}-{token}.full.out is missing", prefix.display()));
        }
        (token.clone(), run)
    });
    let fmm3d: Vec<Vec<Fmm3dRun>> = arguments
        .fmm3d
        .iter()
        .map(|(_, prefix)| {
            arguments
                .eps
                .iter()
                .map(|e| read_run(prefix, e, &problem))
                .collect()
        })
        .collect();

    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    let root = world.rank() == 0;
    let oracle = Oracle::sharded(
        complex_k(&case),
        &problem,
        true,
        arguments.oracle_threads.unwrap_or(arguments.threads),
        &world,
    );
    let kd = if arguments.kim == 0.0 {
        format!("{}", arguments.kd)
    } else {
        format!("{} + {}i", arguments.kd, arguments.kim)
    };
    if root {
        println!(
            "# Helmholtz FMM beside FMM3D: {} at kD = {kd}",
            workload.name()
        );
        println!();
        println!(
            "Machine: {}, {}, {}; ours on {} ranks × {} threads; BLAS threads: \
             OPENBLAS_NUM_THREADS={}; N = {}, {} sampled targets, one {} charge vector",
            cpu_model(),
            cores(),
            target(),
            world.size(),
            arguments.threads,
            std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
            problem.len(),
            problem.sample.len(),
            match case.charges {
                ChargeKind::Real => "real",
                ChargeKind::Complex => "complex",
            },
        );
        if let Some((token, run)) = &reference {
            let e = sample_errors(&run.sampled, &oracle, 0);
            let time = run
                .seconds
                .map_or("–".into(), |(s, t)| format!("{s:.2} s on {t} threads"));
            println!(
                "Reference: FMM3D {} at eps = {token}, against the direct sum at the samples: \
                 φ {:.2e}, ∇φ {:.2e}; {time}",
                arguments.fmm3d[0].0, e[0], e[1]
            );
        }
        println!();
        println!(
            "| Workload | N | kD | ε | code | threads | p by level | build s | run s | φ | ∇φ \
             | all φ | all ∇φ | worst box φ / ε, ∇φ / ε | worst box / all (φ / ∇φ) of boxes | checksum |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: \
             | ---: | --- | --- | --- |"
        );
    }
    for (j, token) in arguments.eps.iter().enumerate() {
        let eps: f64 = token.parse().expect("checked");
        case.degrees = Degrees::Tolerance(eps);
        let ours = run_output(&case, &problem, &oracle, arguments.threads, &world)
            .unwrap_or_else(|e| fail(format!("our FMM at ε = {token}: {e}")));
        let row = &ours.row;
        let grid = Grid {
            domain: ours.domain,
            level: row.levels - 1,
        };
        let block = problem.block(world.rank() as usize, world.size() as usize);
        let check = reference.as_ref().map(|(_, r)| {
            crosscheck(
                &ours.output,
                0,
                &problem,
                &block,
                r.full.as_ref().expect("checked"),
                &grid,
                &world,
            )
        });
        if !root {
            continue;
        }
        if let Some(c) = &check {
            for (j, field) in ["φ", "∇φ"].into_iter().enumerate() {
                eprintln!(
                    "{} kD = {kd} ε = {token}: ours, worst boxes of {field} (box, error over the ℓ2 error): {}",
                    workload.name(),
                    worst_boxes(c, j, 5)
                );
            }
        }
        let errors = row.errors.last().expect("an entry over all vectors");
        println!(
            "| {} | {} | {kd} | {token} | ours ({}) | {} × {} | {} | {:.2} | {:.2} | {:.2e} | {:.2e} | {} | {:016x} |",
            workload.name(),
            problem.len(),
            row.p2p,
            row.ranks,
            row.threads,
            degrees_string(&row.degrees),
            row.build,
            row.evaluate,
            errors[0],
            errors[1],
            crosscheck_cells(check.as_ref(), eps),
            row.checksum,
        );
        for ((label, _), runs) in arguments.fmm3d.iter().zip(&fmm3d) {
            let run = &runs[j];
            let e = sample_errors(&run.sampled, &oracle, 0);
            let check = match (&reference, &run.full) {
                (Some((_, r)), Some(full)) => Some(crosscheck_values(
                    full,
                    r.full.as_ref().expect("checked"),
                    &problem.points,
                    &grid,
                )),
                _ => None,
            };
            if let Some(c) = &check {
                for (j, field) in ["φ", "∇φ"].into_iter().enumerate() {
                    eprintln!(
                        "{} kD = {kd} ε = {token}: FMM3D {label}, worst boxes of {field}: {}",
                        workload.name(),
                        worst_boxes(c, j, 5)
                    );
                }
            }
            let (time, threads) = run.seconds.map_or(("–".into(), "–".into()), |(s, t)| {
                (format!("{s:.2}"), t.to_string())
            });
            println!(
                "| {} | {} | {kd} | {token} | FMM3D {label} | {threads} | eps = {token} | – | {time} | {:.2e} | {:.2e} | {} | – |",
                workload.name(),
                problem.len(),
                e[0],
                e[1],
                crosscheck_cells(check.as_ref(), eps),
            );
        }
        eprintln!(
            "{} kD = {kd} ε = {token}: levels {}, leaf level {}, {} leaves, boxes of level {} for the crosscheck",
            workload.name(),
            row.levels,
            row.min_leaf_level,
            row.leaves,
            grid.level
        );
    }
}

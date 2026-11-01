//! The calibration of the Helmholtz FMM's per-level degree p_l over (k a_l, ε) in the full
//! FMM (Helmholtz CH3.3; docs/helmholtz/tasks/CH3.3.md; `nd_fmm_validate::
//! helmholtz_calibration`). Prints Markdown on stdout.
//!
//! Two modes:
//!
//! - **Scans** (MPI, any number of ranks): per kD the workload's problem and oracle are
//!   built once (CH3.1c's harness, `helmholtz_accuracy`), then the FMM runs at consecutive
//!   degrees until the larger of the errors of φ and ∇φ meets half the smallest ε at three
//!   degrees in a row, or the degree reaches `--max-degree`:
//!   - `--scan plateau`: one degree p on every level ≥ 2, from `--first` (default 1);
//!   - `--scan levels`: each level l ≥ 2 of the tree with x_l = Re(k) a_l ≥ `--min-x`
//!     (default 0) in turn, its degree from max(1, ⌊x_l⌋ − 2), every other level held at
//!     `s7_degree` at `--hold-eps` (default the smallest ε), S7's per-pair upper bound,
//!     whose errors lie far below ε (CH3.1c); or, with `--hold α,β,C`, at the
//!     `DegreeRule` of those constants at `--hold-eps` (cheaper on large trees, where S7's
//!     plateau makes every run long).
//!
//!   Each scan prints its curve, then per ε the smallest degree meeting ε/2 and ε beside
//!   S7's degree at the same x, and a line `point: <kind> <workload> <kd> <level> <x> <ε>
//!   <p at ε/2> <p at ε> <S7>` per ε for the fit (the kind `plateau` with x the largest
//!   x of the scanned levels, or `level`). A degree that meets ε/2 already at the scan's
//!   first degree is printed as `≤ p`, and its point is not written.
//! - **`fit <files…>`** (no MPI): reads the `point:` lines of saved scan outputs, fits
//!   `DegreeRule` (the plateau's α and β to the points whose x is at most `PLATEAU_X`, of
//!   either scan, C to every point), and prints the rule and each point against it and
//!   against S7.
//!
//! Options of the scans: `--workload w` (as `helmholtz_accuracy`: `w1`, `w2`, `w3`,
//! `w3:<n>`, `w4-<distribution>:<n>`), `--kd x,…`, `--kim y`, `--eps e,…` (default
//! 1e-3,…,1e-8 by decades), `--scan plateau|levels`, `--first p`, `--max-degree p`
//! (default 80), `--min-x x`, `--hold-eps e`, `--hold α,β,C`, `--samples n`, `--threads t` (default 1),
//! `--charges real|complex`. Run in release mode with one BLAS thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_calibrate -- \
//!     --workload w4-cube:100000 --kd 1 --scan plateau --threads 6 > cube.md
//! cargo run --release -p nd-fmm-validate --example helmholtz_calibrate -- fit cube.md
//! ```
//!
//! MPI is initialised with `Threading::Funneled`. Timings are wall times, never asserted.

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::helmholtz::{ChargeKind, DegreeRule, Degrees, Wavenumber, s7_degree};
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{Case, Oracle, Problem, complex_k, run};
use nd_fmm_validate::helmholtz_calibration::{Curve, Point, covers, fit};
use nd_fmm_validate::helmholtz_workloads::Workload;

/// Plateau points with x at most this go to the plateau's fit (S7: p is flat in k a below
/// about 4, `spikes/truncation/SPIKE_REPORT.md` §3).
const PLATEAU_X: f64 = 4.0;

/// The degrees of levels 0..`LEVELS` passed to the FMM (more than any tree has).
const LEVELS: usize = 64;

/// The scan.
#[derive(Clone, Copy, PartialEq)]
enum Scan {
    Plateau,
    Levels,
}

/// The command line of the scans.
struct Arguments {
    workload: String,
    kds: Vec<f64>,
    kim: f64,
    epsilons: Vec<f64>,
    scan: Scan,
    first: usize,
    max_degree: usize,
    min_x: f64,
    hold_epsilon: Option<f64>,
    hold: Option<DegreeRule>,
    samples: usize,
    threads: usize,
    charges: ChargeKind,
}

/// Parses a comma-separated list.
fn list<T: std::str::FromStr>(value: &str) -> Option<Vec<T>> {
    value.split(',').map(|v| v.parse().ok()).collect()
}

/// The workload of a name at kD (W1 and W3 resolved depend on it), as `helmholtz_accuracy`.
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

/// Parses the options of the scans; exits with a message on anything else.
fn arguments(args: &[String]) -> Arguments {
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_calibrate [--workload w] [--kd x,…] [--kim y] [--eps e,…] \
             [--scan plateau|levels] [--first p] [--max-degree p] [--min-x x] \
             [--hold-eps e] [--hold α,β,C] [--samples n] [--threads t] [--charges real|complex], \
             or helmholtz_calibrate fit <files…>; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        workload: "w1".into(),
        kds: vec![10.0],
        kim: 0.0,
        epsilons: (3..=8).map(|d| 10f64.powi(-d)).collect(),
        scan: Scan::Plateau,
        first: 1,
        max_degree: 80,
        min_x: 0.0,
        hold_epsilon: None,
        hold: None,
        samples: 1000,
        threads: 1,
        charges: ChargeKind::Complex,
    };
    for pair in args.chunks(2) {
        let [flag, value] = pair else { usage() };
        let ok = match flag.as_str() {
            "--workload" => {
                parsed.workload = value.clone();
                workload(value, 1.0).is_some()
            }
            "--kd" => list(value).map(|v| parsed.kds = v).is_some(),
            "--kim" => value.parse().map(|v| parsed.kim = v).is_ok(),
            "--eps" => list(value).map(|v| parsed.epsilons = v).is_some(),
            "--scan" => match value.as_str() {
                "plateau" => Some(Scan::Plateau),
                "levels" => Some(Scan::Levels),
                _ => None,
            }
            .map(|s| parsed.scan = s)
            .is_some(),
            "--first" => value.parse().map(|v| parsed.first = v).is_ok(),
            "--max-degree" => value.parse().map(|v| parsed.max_degree = v).is_ok(),
            "--min-x" => value.parse().map(|v| parsed.min_x = v).is_ok(),
            "--hold-eps" => value.parse().map(|v| parsed.hold_epsilon = Some(v)).is_ok(),
            "--hold" => match list::<f64>(value).as_deref() {
                Some(&[alpha, beta, c]) => Some(DegreeRule { alpha, beta, c }),
                _ => None,
            }
            .map(|r| parsed.hold = Some(r))
            .is_some(),
            "--samples" => value.parse().map(|v| parsed.samples = v).is_ok(),
            "--threads" => value.parse().map(|v| parsed.threads = v).is_ok(),
            "--charges" => match value.as_str() {
                "real" => Some(ChargeKind::Real),
                "complex" => Some(ChargeKind::Complex),
                _ => None,
            }
            .map(|c| parsed.charges = c)
            .is_some(),
            _ => false,
        };
        if !ok {
            usage();
        }
    }
    let valid = |e: &f64| *e > 0.0 && *e < 1.0;
    if parsed.kds.is_empty()
        || parsed.epsilons.is_empty()
        || !parsed.epsilons.iter().all(valid)
        || !parsed.hold_epsilon.is_none_or(|e| valid(&e))
        || parsed.threads == 0
        || parsed.first == 0
    {
        usage();
    }
    parsed
}

/// A degree with "≤" when it is the scan's first.
fn shown(degree: Option<usize>, first: usize) -> String {
    match degree {
        Some(p) if p == first => format!("≤ {p}"),
        Some(p) => p.to_string(),
        None => "–".into(),
    }
}

/// One scan's context, for its rows and points.
struct Scanned<'a> {
    kind: &'a str,
    workload: String,
    kd: String,
    level: usize,
    x: f64,
}

/// Prints a curve's thresholds per ε and its `point:` lines.
fn report(arguments: &Arguments, scanned: &Scanned<'_>, curve: &Curve, k: &Wavenumber, a: f64) {
    println!();
    println!(
        "| ε | p at ε/2 | p at ε | S7 at x = {:.3} | φ, ∇φ at p(ε/2) |",
        scanned.x
    );
    println!("| --- | ---: | ---: | ---: | --- |");
    for &epsilon in &arguments.epsilons {
        let half = curve.smallest_degree(epsilon / 2.0);
        let full = curve.smallest_degree(epsilon);
        let s7 = s7_degree(k, a, epsilon);
        let errors = half.map_or("–".into(), |p| {
            let e = curve.errors[p - curve.first];
            format!("{:.2e}, {:.2e}", e[0], e[1])
        });
        println!(
            "| {epsilon:e} | {} | {} | {s7} | {errors} |",
            shown(half, curve.first),
            shown(full, curve.first)
        );
    }
    for &epsilon in &arguments.epsilons {
        let (half, full) = (
            curve.smallest_degree(epsilon / 2.0),
            curve.smallest_degree(epsilon),
        );
        if let (Some(half), Some(full)) = (half, full)
            && half > curve.first
        {
            println!(
                "point: {} {} {} {} {} {epsilon:e} {half} {full} {}",
                scanned.kind,
                scanned.workload.replace(' ', "_"),
                scanned.kd,
                scanned.level,
                scanned.x,
                s7_degree(k, a, epsilon),
            );
        }
    }
}

/// The scans of the module documentation.
fn scans(arguments: &Arguments) {
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    let root = world.rank() == 0;
    let smallest = arguments
        .epsilons
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let hold = arguments.hold_epsilon.unwrap_or(smallest);
    let goal = smallest / 2.0;
    if root {
        println!("# Helmholtz degree calibration");
        println!();
        println!(
            "Machine: {}, {}, {}; {} ranks × {} threads; BLAS threads: \
             OPENBLAS_NUM_THREADS={}",
            cpu_model(),
            cores(),
            target(),
            world.size(),
            arguments.threads,
            std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        );
    }
    for &kd in &arguments.kds {
        let workload = workload(&arguments.workload, kd).expect("checked");
        let k = Wavenumber::new(kd, arguments.kim).expect("a valid wavenumber");
        let mut case = Case::new(workload, (kd, arguments.kim), Degrees::PerLevel(vec![]));
        case.charges = arguments.charges;
        case.samples = arguments.samples;
        let problem = Problem::new(&case);
        let oracle = Oracle::sharded(
            complex_k(&case),
            &problem,
            case.gradients,
            arguments.threads,
            &world,
        );
        let kd_string = if arguments.kim == 0.0 {
            format!("{kd}")
        } else {
            format!("{kd}+{}i", arguments.kim)
        };
        // One run at the first degree on every level for the tree's levels and radii.
        let probe = |degrees: Vec<usize>, case: &mut Case| {
            case.degrees = Degrees::PerLevel(degrees);
            run(case, &problem, &oracle, arguments.threads, &world)
                .unwrap_or_else(|e| panic!("{} at kD = {kd_string}: {e}", workload.name()))
        };
        let plateau_degrees = |p: usize| {
            let mut d = vec![p; LEVELS];
            d[..2].fill(0);
            d
        };
        let first = probe(plateau_degrees(arguments.first), &mut case);
        let radii = first.circumradii.clone();
        let nlevels = first.levels;
        let x_of = |l: usize| k.re() * radii[l];
        // (level, x, a, the degrees at p)
        type DegreesAt = Box<dyn Fn(usize) -> Vec<usize>>;
        let mut jobs: Vec<(usize, f64, f64, usize, DegreesAt)> = Vec::new();
        match arguments.scan {
            Scan::Plateau => {
                if nlevels > 2 {
                    jobs.push((
                        2,
                        x_of(2),
                        radii[2],
                        arguments.first,
                        Box::new(plateau_degrees),
                    ));
                }
            }
            Scan::Levels => {
                let held: Vec<usize> = (0..LEVELS)
                    .map(|l| {
                        if l < 2 || l >= nlevels {
                            0
                        } else if let Some(rule) = &arguments.hold {
                            rule.degree(x_of(l), hold)
                        } else {
                            s7_degree(&k, radii[l], hold)
                        }
                    })
                    .collect();
                for l in 2..nlevels {
                    let x = x_of(l);
                    if x < arguments.min_x {
                        continue;
                    }
                    let held = held.clone();
                    let start = (x.floor() as usize).saturating_sub(2).max(1);
                    jobs.push((
                        l,
                        x,
                        radii[l],
                        start,
                        Box::new(move |p| {
                            let mut d = held.clone();
                            d[l] = p;
                            d
                        }),
                    ));
                }
            }
        }
        for (level, x, a, start, degrees) in jobs {
            let scanned = Scanned {
                kind: if arguments.scan == Scan::Plateau {
                    "plateau"
                } else {
                    "level"
                },
                workload: workload.name(),
                kd: kd_string.clone(),
                level,
                x,
            };
            if root {
                println!();
                println!(
                    "## {} kD = {kd_string}, N = {}: {} {}",
                    scanned.workload,
                    problem.len(),
                    if arguments.scan == Scan::Plateau {
                        "plateau, levels ≥ 2".to_string()
                    } else {
                        format!(
                            "level {level}, others at {} at ε = {hold:e}",
                            arguments.hold.map_or("S7".into(), |r| format!(
                                "α = {}, β = {}, C = {}",
                                r.alpha, r.beta, r.c
                            ))
                        )
                    },
                    format_args!("(x = {x:.3}, {nlevels} levels)"),
                );
                println!();
                println!("| p | φ | ∇φ | evaluate s |");
                println!("| ---: | ---: | ---: | ---: |");
            }
            let mut curve = Curve::new(start);
            while !curve.reached(goal) && curve.next_degree() <= arguments.max_degree {
                let p = curve.next_degree();
                let row = probe(degrees(p), &mut case);
                let all = *row.errors.last().expect("an entry over all vectors");
                if root {
                    println!(
                        "| {p} | {:.2e} | {:.2e} | {:.2} |",
                        all[0], all[1], row.evaluate
                    );
                }
                curve.push(all);
            }
            if root {
                report(arguments, &scanned, &curve, &k, a);
            }
        }
    }
}

/// The `fit` mode of the module documentation.
fn fit_files(files: &[String]) {
    let mut plateau = Vec::new();
    let mut others = Vec::new();
    let mut lines = Vec::new();
    for file in files {
        let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
        for line in text.lines().filter_map(|l| l.strip_prefix("point: ")) {
            let f: Vec<&str> = line.split_whitespace().collect();
            let [kind, workload, kd, level, x, epsilon, half, full, s7] = f[..] else {
                panic!("{file}: a malformed point line: {line}");
            };
            let parse = |v: &str| v.parse::<f64>().unwrap_or_else(|_| panic!("{line}"));
            let point = Point {
                x: parse(x),
                epsilon: parse(epsilon),
                degree: parse(half) as usize,
            };
            if point.x <= PLATEAU_X {
                plateau.push(point);
            } else {
                others.push(point);
            }
            lines.push((
                point,
                format!("{kind} | {workload} | {kd} | {level}"),
                parse(full) as usize,
                parse(s7) as usize,
            ));
        }
    }
    println!("# Helmholtz degree calibration: the fit");
    println!();
    println!(
        "{} plateau points (x ≤ {PLATEAU_X}), {} others, from {}",
        plateau.len(),
        others.len(),
        files.join(", ")
    );
    let Some(rule) = fit(&plateau, &others) else {
        println!("no rule fits");
        std::process::exit(1);
    };
    println!();
    println!(
        "Rule: α = {:.2}, β = {:.2}, C = {:.2}: p = max(⌈{:.2} log₁₀(1/ε) {:+.2}⌉, \
         ⌈x + {:.2} (ln 1/ε)^(2/3) x^(1/3)⌉)",
        rule.alpha, rule.beta, rule.c, rule.alpha, rule.beta, rule.c
    );
    let all: Vec<Point> = plateau.iter().chain(&others).copied().collect();
    assert!(covers(&rule, &all));
    println!();
    println!("| kind | workload | kD | level | x | ε | p at ε/2 | p at ε | rule | S7 |");
    println!("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |");
    for (point, name, full, s7) in &lines {
        println!(
            "| {name} | {:.3} | {:e} | {} | {full} | {} | {s7} |",
            point.x,
            point.epsilon,
            point.degree,
            rule.degree(point.x, point.epsilon)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.split_first() {
        Some((mode, files)) if mode == "fit" && !files.is_empty() => fit_files(files),
        _ => scans(&arguments(&args)),
    }
}

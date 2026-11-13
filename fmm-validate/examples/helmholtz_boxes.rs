//! Where the Helmholtz FMM's errors concentrate (Helmholtz CH4.0b;
//! docs/helmholtz/tasks/CH4.0b.md). Prints Markdown on stdout.
//!
//! ```text
//! helmholtz_boxes [CASE] [--eps e,…] [--m2l rotation|default,…] [--study single|levels]
//!     [--margin m] [--offset x,y,z] [--leaf-size n] [--raise-factor f] [--worst n]
//!     [--per-box s] [--full-max n] [--threads t]
//! ```
//!
//! One case on one rank, against the direct sum (`nd_fmm_validate::helmholtz_boxes`): at
//! every point when N ≤ `--full-max` (default 200,000), else at §9.1's samples plus
//! `--per-box` targets (default 2) per box of the FMM's deepest level. The FMM runs at each
//! ε of `--eps` (default 10⁻⁶) under each M2L of `--m2l` (`rotation`, the default here, or
//! `default`, the library's `Auto`), the first pair first. Per run: the degrees, §9.1's
//! relative ℓ2 errors at the 1,000 samples, the relative ℓ2 errors over the reference's
//! targets, the box measure of the deepest level (the largest error at one of a box's targets
//! over the reference's RMS) over the run's ε and over the ℓ2 error, the boxes above ε, the
//! build and the evaluation. For the first run: the `--worst` worst boxes of ∇φ and of φ
//! with their place (the box's cell and centre, whether it touches the domain's boundary, its
//! worst target's distance from its box's centre on every level from 2 in units of the
//! half-width, √3 at a corner), and the errors by the number of levels on which a target lies
//! within 0.1 of a face.
//!
//! `--study levels` also runs the first case once per level with M2L with that level's
//! degree raised to the rule's at ε × `--raise-factor` (default 10⁻³), and once with every
//! level raised; each level's part of the error is the first run's error minus the raised
//! run's (`helmholtz_boxes::level_shares`), over every target, at the worst boxes' worst
//! targets and by the target's distance from its box's centre on that level. Named degrees
//! change the exponential M2L's precision rule, so the study is read on `--m2l rotation`.
//!
//! CASE is `--workload w` (as `helmholtz_fmm3d`: `w1`, `w2`, `w3`, `w3:<n>`,
//! `w4-<distribution>:<n>`; default `w1`), `--kd x` (default 30), `--kim y` (default 0),
//! `--density random|planewave|constant` (default `random`, §9.1's) and `--charges
//! real|complex` (default complex). `--margin m` and `--offset x,y,z` set the FMM's domain to
//! [−m, 1 + m]³ shifted by the offset (default: the points' bounding cube); `--leaf-size n`
//! the points per leaf (`uniform` for N + 1: leaves on the leaf rule's level alone, no W or X
//! list).
//!
//! Run in release mode on one rank (`mpirun -n 1`), with one BLAS thread. Accuracy only; the
//! times are wall times, reported, never asserted.

use std::time::Instant;

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::helmholtz::{ChargeKind, Degrees, HelmholtzM2l, Wavenumber};
use nd_fmm_validate::bench::{cores, cpu_model};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    Case, Density, Oracle, Problem, Run, complex_k, run_output,
};
use nd_fmm_validate::helmholtz_boxes::{
    BoxError, Errors, Reference, box_sample, cell, face_closeness, level_parts_at, level_shares,
    offsets, worst_boxes,
};
use nd_fmm_validate::helmholtz_fmm3d::Grid;
use nd_fmm_validate::helmholtz_workloads::{SEED, Workload};

/// The command line.
struct Arguments {
    workload: String,
    kd: f64,
    kim: f64,
    density: Density,
    charges: ChargeKind,
    eps: Vec<f64>,
    default_m2l: Vec<bool>,
    levels: bool,
    margin: Option<f64>,
    offset: [f64; 3],
    leaf_size: Option<String>,
    raise_factor: f64,
    worst: usize,
    per_box: usize,
    full_max: usize,
    threads: usize,
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
            "usage: helmholtz_boxes [--workload w] [--kd x] [--kim y] \
             [--density random|planewave|constant] [--charges real|complex] [--eps e] \
             [--m2l rotation|default] [--study single|levels] [--margin m] [--offset x,y,z] \
             [--leaf-size n|uniform] [--raise-factor f] [--worst n] [--per-box s] \
             [--full-max n] [--threads t]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        workload: "w1".into(),
        kd: 30.0,
        kim: 0.0,
        density: Density::Random,
        charges: ChargeKind::Complex,
        eps: vec![1e-6],
        default_m2l: vec![false],
        levels: false,
        margin: None,
        offset: [0.0; 3],
        leaf_size: None,
        raise_factor: 1e-3,
        worst: 8,
        per_box: 2,
        full_max: 200_000,
        threads: 1,
    };
    let mut rest = &args[..];
    while let [flag, value, tail @ ..] = rest {
        let ok = match flag.as_str() {
            "--workload" => {
                parsed.workload = value.clone();
                workload(value, 1.0).is_some()
            }
            "--kd" => value.parse().map(|v| parsed.kd = v).is_ok(),
            "--kim" => value.parse().map(|v| parsed.kim = v).is_ok(),
            "--density" => Density::from_name(value)
                .map(|d| parsed.density = d)
                .is_some(),
            "--charges" => match value.as_str() {
                "real" => Some(ChargeKind::Real),
                "complex" => Some(ChargeKind::Complex),
                _ => None,
            }
            .map(|c| parsed.charges = c)
            .is_some(),
            "--eps" => value
                .split(',')
                .map(|v| v.parse::<f64>().ok().filter(|&e| e > 0.0 && e < 1.0))
                .collect::<Option<Vec<f64>>>()
                .map(|e| parsed.eps = e)
                .is_some(),
            "--m2l" => value
                .split(',')
                .map(|v| match v {
                    "rotation" => Some(false),
                    "default" => Some(true),
                    _ => None,
                })
                .collect::<Option<Vec<bool>>>()
                .map(|d| parsed.default_m2l = d)
                .is_some(),
            "--study" => match value.as_str() {
                "single" => Some(false),
                "levels" => Some(true),
                _ => None,
            }
            .map(|l| parsed.levels = l)
            .is_some(),
            "--margin" => value
                .parse::<f64>()
                .ok()
                .filter(|&m| m >= 0.0)
                .map(|m| parsed.margin = Some(m))
                .is_some(),
            "--offset" => {
                let parts: Option<Vec<f64>> = value.split(',').map(|v| v.parse().ok()).collect();
                parts
                    .filter(|p| p.len() == 3)
                    .map(|p| parsed.offset = [p[0], p[1], p[2]])
                    .is_some()
            }
            "--leaf-size" => {
                parsed.leaf_size = Some(value.clone());
                value == "uniform" || value.parse::<usize>().is_ok_and(|n| n > 0)
            }
            "--raise-factor" => value
                .parse::<f64>()
                .ok()
                .filter(|&f| f > 0.0 && f < 1.0)
                .map(|f| parsed.raise_factor = f)
                .is_some(),
            "--worst" => value.parse().map(|v| parsed.worst = v).is_ok(),
            "--per-box" => value.parse().map(|v| parsed.per_box = v).is_ok(),
            "--full-max" => value.parse().map(|v| parsed.full_max = v).is_ok(),
            "--threads" => value.parse().map(|v| parsed.threads = v).is_ok(),
            _ => false,
        };
        if !ok {
            usage();
        }
        rest = tail;
    }
    if !rest.is_empty() || parsed.threads == 0 {
        usage();
    }
    if parsed.density == Density::PlaneWave && parsed.charges == ChargeKind::Real {
        eprintln!("helmholtz_boxes: a plane wave's trace needs complex charges");
        std::process::exit(2);
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

/// One measured run.
struct Measured {
    label: String,
    eps: f64,
    run: Run,
    errors: Errors,
    boxes: Vec<BoxError>,
}

/// Runs `case` and measures it against `reference` on `grid` (the first run's deepest level,
/// or this run's when `None`).
#[allow(clippy::too_many_arguments)]
fn measure<C: Communicator>(
    label: String,
    eps: f64,
    case: &Case,
    problem: &Problem,
    oracle: &Oracle,
    reference: &Reference,
    grid: Option<&Grid>,
    threads: usize,
    world: &C,
) -> Measured {
    let start = Instant::now();
    let run = run_output(case, problem, oracle, threads, world).unwrap_or_else(|e| {
        eprintln!("helmholtz_boxes: {label}: {e}");
        std::process::exit(1);
    });
    let fmm = start.elapsed().as_secs_f64();
    let errors = Errors::of(&run.output, 0, reference);
    let own = Grid {
        domain: run.domain,
        level: run.row.levels - 1,
    };
    let boxes = errors.boxes(reference, &problem.points, grid.unwrap_or(&own));
    // The run as it finishes (the tables print after the last run): §9.1's errors and those
    // at the reference's targets, the build, the evaluation and its M2L by level, the rest
    // of `run_output` (the samples' errors) and the errors at the reference's targets.
    let row = &run.row;
    let m2l: Vec<String> = row
        .m2l_seconds
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s > 0.0)
        .map(|(l, s)| format!("{l}: {s:.1}"))
        .collect();
    let all = errors.l2(reference);
    eprintln!(
        "helmholtz_boxes: {label}: p {}; §9.1 φ, ∇φ {:.2e}, {:.2e}, all {:.2e}, {:.2e}; \
         build {:.1} s, evaluate {:.1} s (M2L by level {}), the rest of the run {:.1} s, the \
         errors at {} targets {:.1} s",
        degrees_string(&row.degrees),
        row.errors[0][0],
        row.errors[0][1],
        all[0],
        all[1],
        row.build,
        row.evaluate,
        m2l.join(", "),
        fmm - row.build - row.evaluate,
        reference.targets.len(),
        start.elapsed().as_secs_f64() - fmm,
    );
    Measured {
        label,
        eps,
        run,
        errors,
        boxes,
    }
}

/// The row of one run in the runs table.
fn run_row(m: &Measured, reference: &Reference) {
    let (row, eps) = (&m.run.row, m.eps);
    let l2 = m.errors.l2(reference);
    let worst = m.boxes.iter().fold([0.0f64; 2], |w, b| {
        [w[0].max(b.error[0]), w[1].max(b.error[1])]
    });
    let above = |f: usize| m.boxes.iter().filter(|b| b.error[f] > eps).count();
    println!(
        "| {} | {} | {} | {:.2e}, {:.2e} | {:.2e}, {:.2e} | {:.2}, {:.2} | {:.0}, {:.0} | {}, {} of {} | {:.2}, {:.2} |",
        m.label,
        degrees_string(&row.degrees),
        row.leaves,
        row.errors[0][0],
        row.errors[0][1],
        l2[0],
        l2[1],
        worst[0] / eps,
        worst[1] / eps,
        worst[0] / l2[0].max(f64::MIN_POSITIVE),
        worst[1] / l2[1].max(f64::MIN_POSITIVE),
        above(0),
        above(1),
        m.boxes.len(),
        row.build,
        row.evaluate,
    );
}

/// Prints the worst boxes of field `field` of `base` with their place, and with the levels'
/// parts at their worst targets when `raised` is given.
#[allow(clippy::too_many_arguments)]
fn worst_table(
    base: &Measured,
    raised: Option<&[Option<Errors>]>,
    reference: &Reference,
    problem: &Problem,
    grid: &Grid,
    field: usize,
    count: usize,
    eps: f64,
) {
    let name = ["φ", "∇φ"][field];
    let deepest = grid.level;
    let cells = 1u64 << deepest;
    println!();
    println!(
        "Worst boxes of {name} (level {deepest}): the cell, its centre in the domain's unit \
         cube, the error over ε, the targets, whether a face lies on the domain's boundary, the \
         worst target's distance |v| from its box's centre on levels 2–{deepest} over the \
         half-width (√3 at a corner){}.",
        if raised.is_some() {
            ", and each level's part of its error"
        } else {
            ""
        }
    );
    println!();
    println!(
        "| # | cell | centre | error / ε | targets | boundary | |v| / r, levels 2–{deepest} | level parts |"
    );
    println!("| ---: | --- | --- | ---: | ---: | --- | --- | --- |");
    for (rank, b) in worst_boxes(&base.boxes, field, count).iter().enumerate() {
        let c = cell(b.key);
        let centre: Vec<String> = c
            .iter()
            .map(|&i| format!("{:.3}", (i as f64 + 0.5) / cells as f64))
            .collect();
        let boundary = c.iter().any(|&i| i == 0 || i == cells - 1);
        let j = b.worst[field];
        let x = problem.points[reference.targets[j]];
        let profile: Vec<String> = offsets(x, grid, deepest)
            .iter()
            .skip(2)
            .map(|o| format!("{:.2}", o.iter().map(|v| v * v).sum::<f64>().sqrt()))
            .collect();
        let parts = raised.map_or(String::new(), |raised| {
            level_parts_at(&base.errors, raised, j)
                .iter()
                .enumerate()
                .filter(|(_, p)| p[field] >= 0.1)
                .map(|(l, p)| format!("{l}: {:.2}", p[field]))
                .collect::<Vec<_>>()
                .join(", ")
        });
        println!(
            "| {} | ({}, {}, {}) | ({}) | {:.2} | {} | {} | {} | {} |",
            rank + 1,
            c[0],
            c[1],
            c[2],
            centre.join(", "),
            b.error[field] / eps,
            b.targets,
            if boundary { "yes" } else { "no" },
            profile.join(" "),
            parts,
        );
    }
}

/// The errors by the number of levels 2..=deepest on which a target lies within 0.1 of a
/// face of its box.
fn faces_table(base: &Measured, reference: &Reference, problem: &Problem, grid: &Grid, eps: f64) {
    let deepest = grid.level;
    let rms = reference.rms();
    let mut bins = vec![(0usize, [0.0f64; 2], [0.0f64; 2]); deepest.max(1)];
    for (j, &i) in reference.targets.iter().enumerate() {
        let closeness = face_closeness(&offsets(problem.points[i], grid, deepest));
        let faces = closeness.iter().skip(2).filter(|&&v| v >= 0.9).count();
        let e = base.errors.norms(j);
        let bin = &mut bins[faces.min(deepest.saturating_sub(1))];
        bin.0 += 1;
        for (f, e) in e.into_iter().enumerate() {
            bin.1[f] += e * e;
            bin.2[f] = bin.2[f].max(e);
        }
    }
    println!();
    println!(
        "Errors by the levels (of 2–{deepest}) on which the target lies within 0.1 of a face of \
         its box: the RMS error over the reference's RMS, and the largest, over ε."
    );
    println!();
    println!("| levels on a face | targets | RMS φ, ∇φ / ε | largest φ, ∇φ / ε |");
    println!("| ---: | ---: | --- | --- |");
    for (faces, (n, sums, max)) in bins.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let r = |f: usize| (sums[f] / *n as f64).sqrt() / rms[f] / eps;
        println!(
            "| {faces} | {n} | {:.3}, {:.3} | {:.2}, {:.2} |",
            r(0),
            r(1),
            max[0] / rms[0] / eps,
            max[1] / rms[1] / eps,
        );
    }
}

fn main() {
    let args = arguments();
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    if world.size() != 1 {
        eprintln!("helmholtz_boxes: one rank only");
        std::process::exit(2);
    }
    let workload = workload(&args.workload, args.kd).expect("checked");
    let (eps, default_m2l) = (args.eps[0], args.default_m2l[0]);
    let m2l_of = |default: bool| {
        if default {
            HelmholtzM2l::Auto
        } else {
            HelmholtzM2l::Rotation
        }
    };
    let mut case = Case::new(workload, (args.kd, args.kim), Degrees::Tolerance(eps));
    case.charges = args.charges;
    case.density = args.density;
    case.m2l = m2l_of(default_m2l);
    let setup = Instant::now();
    let problem = Problem::new(&case);
    let n = problem.len();
    if let Some(margin) = args.margin {
        let o = args.offset;
        case.domain = Some([
            -margin + o[0],
            -margin + o[1],
            -margin + o[2],
            1.0 + margin + o[0],
            1.0 + margin + o[1],
            1.0 + margin + o[2],
        ]);
    } else if args.offset != [0.0; 3] {
        eprintln!("helmholtz_boxes: --offset needs --margin");
        std::process::exit(2);
    }
    case.max_points_per_leaf = args.leaf_size.as_deref().map(|s| match s {
        "uniform" => n + 1,
        n => n.parse().expect("checked"),
    });
    let k = complex_k(&case);
    let oracle = Oracle::sharded(k, &problem, true, args.threads, &world);
    eprintln!(
        "helmholtz_boxes: the problem and the samples' oracle, {:.1} s",
        setup.elapsed().as_secs_f64()
    );

    let m2l_name = |default: bool| if default { "default" } else { "Rotation" };
    println!(
        "# The box study: {} at kD = {}{}, ε = {:e}, {} density, {}",
        workload.name(),
        args.kd,
        if args.kim != 0.0 {
            format!(" + {}i", args.kim)
        } else {
            String::new()
        },
        eps,
        args.density.name(),
        m2l_name(default_m2l),
    );
    println!();
    println!(
        "Machine: {} ({} cores), 1 rank × {} threads. N = {n}. Domain: {}. Leaf size: {}.",
        cpu_model(),
        cores(),
        args.threads,
        case.domain
            .map_or("the points' bounding cube".into(), |d| format!("{d:?}")),
        args.leaf_size.as_deref().unwrap_or("the library's"),
    );

    // The first run fixes the grid of the box measure, and so the per-box sample.
    let first = measure(
        format!("ε = {eps:e}, {}", m2l_name(default_m2l)),
        eps,
        &case,
        &problem,
        &oracle,
        &Reference::at(k, &problem, 0, problem.sample.clone(), args.threads),
        None,
        args.threads,
        &world,
    );
    let grid = Grid {
        domain: first.run.domain,
        level: first.run.row.levels - 1,
    };
    let start = Instant::now();
    let reference = if n <= args.full_max {
        Reference::every(k, &problem, 0, args.threads)
    } else {
        let targets = box_sample(&problem.points, &problem.sample, &grid, args.per_box, SEED);
        Reference::at(k, &problem, 0, targets, args.threads)
    };
    println!(
        "Reference: the direct sum at {} of {n} targets ({}), {:.1} s.",
        reference.targets.len(),
        if reference.targets.len() == n {
            "every point".to_owned()
        } else {
            format!(
                "§9.1's samples and {} per box of level {}",
                args.per_box, grid.level
            )
        },
        start.elapsed().as_secs_f64()
    );
    let errors = Errors::of(&first.run.output, 0, &reference);
    let base = Measured {
        boxes: errors.boxes(&reference, &problem.points, &grid),
        errors,
        ..first
    };

    let mut runs = Vec::new();
    for &other_eps in &args.eps {
        for &other_m2l in &args.default_m2l {
            if (other_eps, other_m2l) == (eps, default_m2l) {
                continue;
            }
            let other = Case {
                degrees: Degrees::Tolerance(other_eps),
                m2l: m2l_of(other_m2l),
                ..case.clone()
            };
            runs.push(measure(
                format!("ε = {other_eps:e}, {}", m2l_name(other_m2l)),
                other_eps,
                &other,
                &problem,
                &oracle,
                &reference,
                Some(&grid),
                args.threads,
                &world,
            ));
        }
    }
    let mut raised: Option<Vec<Option<Errors>>> = None;
    if args.levels {
        let wavenumber = Wavenumber::new(args.kd, args.kim).expect("a valid wavenumber");
        let tight = Degrees::Tolerance(eps * args.raise_factor)
            .resolve(&wavenumber, &base.run.domain, base.run.row.levels)
            .expect("a valid tolerance");
        let degrees = base.run.row.degrees.clone();
        let mut parts = vec![None; degrees.len()];
        for l in 2..degrees.len() {
            if base.run.row.v_pairs[l] == 0 {
                continue;
            }
            let mut named = degrees.clone();
            named[l] = tight[l].max(degrees[l] + 1);
            let raised_case = Case {
                degrees: Degrees::PerLevel(named),
                ..case.clone()
            };
            let m = measure(
                format!("level {l} at {}", tight[l].max(degrees[l] + 1)),
                eps,
                &raised_case,
                &problem,
                &oracle,
                &reference,
                Some(&grid),
                args.threads,
                &world,
            );
            parts[l] = Some(m.errors.clone());
            runs.push(m);
        }
        let mut all = degrees.clone();
        for l in 2..all.len() {
            all[l] = tight[l].max(degrees[l] + 1);
        }
        let all_case = Case {
            degrees: Degrees::PerLevel(all),
            ..case.clone()
        };
        runs.push(measure(
            "every level raised".into(),
            eps,
            &all_case,
            &problem,
            &oracle,
            &reference,
            Some(&grid),
            args.threads,
            &world,
        ));
        raised = Some(parts);
    }

    println!();
    println!(
        "Runs: the degrees by level; the leaves; §9.1's relative ℓ2 errors at the 1,000 samples; \
         the relative ℓ2 errors over the reference's targets; the worst box of level {} over ε \
         and over the ℓ2 error; the boxes whose error exceeds ε; the build and the evaluation \
         in seconds.",
        grid.level
    );
    println!();
    println!(
        "| run | p by level | leaves | §9.1 φ, ∇φ | all φ, ∇φ | worst box / ε | worst / ℓ2 | boxes > ε φ, ∇φ | build, evaluate s |"
    );
    println!("| --- | --- | ---: | --- | --- | --- | --- | --- | --- |");
    run_row(&base, &reference);
    for m in &runs {
        run_row(m, &reference);
    }

    if let Some(raised) = &raised {
        let shares = level_shares(&base.errors, raised);
        println!();
        println!(
            "Each level's part of the error (the rule's run minus the run with that level \
             raised): its share of Σ|e|² over the reference's targets, and the rest."
        );
        println!();
        println!("| level | p | raised to | V pairs | share φ | share ∇φ |");
        println!("| ---: | ---: | ---: | ---: | ---: | ---: |");
        let row = &base.run.row;
        for (l, share) in shares.iter().enumerate().take(raised.len()) {
            if raised[l].is_none() {
                continue;
            }
            let label = runs
                .iter()
                .find(|m| m.label.starts_with(&format!("level {l} ")))
                .map_or(String::new(), |m| {
                    m.label.rsplit(' ').next().unwrap_or("").into()
                });
            println!(
                "| {l} | {} | {label} | {} | {:.3} | {:.3} |",
                row.degrees[l], row.v_pairs[l], share[0], share[1]
            );
        }
        let rest = shares.last().expect("the rest");
        println!("| rest | | | | {:.3} | {:.3} |", rest[0], rest[1]);
    }

    for field in [1, 0] {
        worst_table(
            &base,
            raised.as_deref(),
            &reference,
            &problem,
            &grid,
            field,
            args.worst,
            eps,
        );
    }
    faces_table(&base, &reference, &problem, &grid, eps);
    if let Some(raised) = &raised {
        distance_table(&base, raised, &reference, &problem, &grid, eps);
    }
}

/// The edges of the bins of a target's distance from its box's centre over the half-width
/// (√3 at a corner).
const DISTANCE_BINS: [f64; 6] = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5];

/// Each level's part of the error by the target's distance from the centre of its box on
/// that level, over the half-width: the RMS of the part over the reference's RMS, over ε.
fn distance_table(
    base: &Measured,
    raised: &[Option<Errors>],
    reference: &Reference,
    problem: &Problem,
    grid: &Grid,
    eps: f64,
) {
    let rms = reference.rms();
    let nbins = DISTANCE_BINS.len();
    println!();
    println!(
        "Each level's part of the ∇φ error (φ's in brackets) by the target's distance |v| from \
         the centre of its box on that level, over the half-width r: the RMS of the part over \
         the reference's RMS, over ε, and the targets per bin."
    );
    println!();
    let heads: Vec<String> = (0..nbins)
        .map(|b| match DISTANCE_BINS.get(b + 1) {
            Some(hi) => format!("{:.2}–{:.2}", DISTANCE_BINS[b], hi),
            None => format!("≥ {:.2}", DISTANCE_BINS[b]),
        })
        .collect();
    println!("| level | {} |", heads.join(" | "));
    println!("| ---: |{}", " ---: |".repeat(nbins));
    for (l, errors) in raised.iter().enumerate() {
        let Some(errors) = errors else { continue };
        let mut bins = vec![(0usize, [0.0f64; 2]); nbins];
        for (j, &i) in reference.targets.iter().enumerate() {
            let o = offsets(problem.points[i], grid, l);
            let distance = o[l].iter().map(|v| v * v).sum::<f64>().sqrt();
            let b = DISTANCE_BINS
                .iter()
                .rposition(|&e| distance >= e)
                .unwrap_or(0);
            let part = Errors {
                values: vec![std::array::from_fn(|c| {
                    base.errors.values[j][c] - errors.values[j][c]
                })],
            }
            .norms(0);
            bins[b].0 += 1;
            for (sum, e) in bins[b].1.iter_mut().zip(part) {
                *sum += e * e;
            }
        }
        let cells: Vec<String> = bins
            .iter()
            .map(|(n, sums)| {
                if *n == 0 {
                    "–".into()
                } else {
                    let r = |f: usize| (sums[f] / *n as f64).sqrt() / rms[f] / eps;
                    format!("{:.3} ({:.3}), {n}", r(1), r(0))
                }
            })
            .collect();
        println!("| {l} | {} |", cells.join(" | "));
    }
}

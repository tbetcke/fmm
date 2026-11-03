//! The Helmholtz FMM against the sampled direct sum on the reference workloads
//! (Helmholtz CH3.1c; docs/design/helmholtz-fmm-proposal.md §9.1 and §10, CH3.1). Prints
//! Markdown on stdout.
//!
//! One case per kD and ε: the workload's points as sources and targets, the degrees of
//! S7's closed form at ε (or named, the same on every level), real or complex charges,
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
//! - `--eps e,…`: the tolerances (default 1e-3); `--degree p` names p on every level
//!   in place of a tolerance (levels 0 and 1 hold expansions nothing reads);
//! - `--charges real|complex` (default complex), `--vectors m` (default 1),
//!   `--samples n` (default 1000), `--no-gradients`;
//! - `--threads t,…` (default 1), `--oracle-threads t` (default the largest of
//!   `--threads`);
//! - `--rule calibrated|s7` (default `calibrated`, Helmholtz CH3.3): with `s7`, a tolerance
//!   gives S7's closed form (`s7_degree`) per level in place of the default rule, through
//!   one run at degree 1 for the levels' circumradii (the times before CH3.3's rule);
//! - `--p2p auto|reference|<isa>,…` (default `auto`): the P2P kernels (Helmholtz CH3.4c),
//!   each case run with each in turn; an ISA this machine lacks is refused before MPI
//!   starts.
//! - `--m2l rotation|dense|classes|batched|batched-fused,…` (default `rotation`, Helmholtz
//!   CH3.2a, CH3.2b, CH3.2c; `batched-fused` the batched M2L in the fused mode): the M2L,
//!   each case run with each in turn (and each P2P kernel);
//!   `--dense-budget MB` the dense, class or batched tables' bytes per rank (default
//!   1024 MiB). The column "M2L" names the levels with tables and their MB. Also (Helmholtz
//!   CH3.2d) `auto`, the rule per level at the rule's crossover, `auto:<d>` at the crossover
//!   d, and `sweep`: Auto at every crossover that chooses differently, from Batched on every
//!   level that fits (crossover ∞) down to Rotation on every level (one below the smallest
//!   p_l), the crossovers after the first from its degrees; the column names the crossover
//!   (`auto ≤ d`). Also (Helmholtz CH6.1e) `planewave`, the exponential M2L with each level's
//!   quadrature by the rule, `planewave:<1|2|3>`, that iprec on every level whose κ_l = k w_l
//!   lies in the quadrature's tables, and `planewave-scan`: Rotation, then each such level
//!   ≥ 2 at iprec 1, 2 and 3 alone, the other levels Rotation; the column names the levels
//!   with waves and their precisions (`planewave 3:i2`).
//! - `--plane-wave-budget MB|whole` (Helmholtz CH6.2a): the exponential M2L's waves buffer
//!   per node (default the library's, `DEFAULT_PLANE_WAVE_NODE_BUDGET`; `whole` a whole
//!   level). The column "M2L" of an exponential-M2L row gives rank 0's waves buffer, its
//!   chunks and level calls and the waves computed (box × frame × vector), and the run ends
//!   with rank 0's peak resident size (Linux).
//!
//! The last column, "÷ Rotation", is the row's error of φ and of ∇φ over the case's
//! Rotation row at the same P2P kernel and threads, where one ran before it.
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
use nd_fmm_exec::helmholtz::{
    ChargeKind, DEFAULT_DENSE_BUDGET, Degrees, HelmholtzM2l, PlaneWavePrecision, Wavenumber,
    s7_degree,
};
use nd_fmm_exec::operator::{MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::domain_of;
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    Case, Oracle, Problem, Row, complex_k, run, sweep_crossovers,
};
use nd_fmm_validate::helmholtz_workloads::Workload;
use nd_fmm_validate::scaling::resident_memory;

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
    p2p: Vec<P2pChoice>,
    m2l: Vec<M2lRun>,
    dense_budget: usize,
    plane_wave_budget: Option<usize>,
    s7: bool,
}

/// One entry of `--m2l`.
#[derive(Clone, Copy, Debug)]
enum M2lRun {
    /// A setting, its multiply–add mode and crossover (`None`: the library's).
    Fixed(HelmholtzM2l, Option<MulAdd>, Option<usize>),
    /// Auto at every crossover that chooses differently (module documentation).
    Sweep,
    /// The exponential M2L, by the rule (`None`) or one precision on every level.
    PlaneWave(Option<Precision>),
    /// Rotation, then each level with a quadrature at each precision alone.
    PlaneWaveScan,
}

/// The levels ≥ 2 of a row whose κ_l = k w_l lies in the quadrature's tables.
fn plane_wave_levels(row: &Row, k: Complex<f64>) -> Vec<usize> {
    (2..row.levels)
        .filter(|&l| {
            // The width w_l = 2 r_l, from the circumradius √3 r_l.
            let width = 2.0 * row.circumradii[l] / 3f64.sqrt();
            domain_of(k * width).is_some()
        })
        .collect()
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
             [--oracle-threads t] [--rule calibrated|s7] [--p2p auto|reference|<isa>,…] \
             [--m2l rotation|dense|classes|batched|batched-fused|auto|auto:<d>|sweep|\
             planewave|planewave:<1|2|3>|planewave-scan,…] \
             [--dense-budget MB] [--plane-wave-budget MB|whole]; got {args:?}"
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
        p2p: vec![P2pChoice::Auto],
        m2l: vec![M2lRun::Fixed(HelmholtzM2l::Rotation, None, None)],
        dense_budget: DEFAULT_DENSE_BUDGET,
        plane_wave_budget: None,
        s7: false,
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
                    "--p2p" => list::<P2pChoice>(value)
                        .filter(|c| c.iter().all(|c| c.resolve().is_ok()))
                        .map(|c| parsed.p2p = c)
                        .is_some(),
                    "--m2l" => value
                        .split(',')
                        .map(|v| {
                            let fixed = |m2l, mul_add| Some(M2lRun::Fixed(m2l, mul_add, None));
                            match v {
                                "rotation" => fixed(HelmholtzM2l::Rotation, None),
                                "dense" => fixed(HelmholtzM2l::Dense, None),
                                "classes" => fixed(HelmholtzM2l::Classes, None),
                                "batched" => fixed(HelmholtzM2l::Batched, Some(MulAdd::Separate)),
                                "batched-fused" => {
                                    fixed(HelmholtzM2l::Batched, Some(MulAdd::Fused))
                                }
                                "auto" => fixed(HelmholtzM2l::Auto, None),
                                "sweep" => Some(M2lRun::Sweep),
                                "planewave" => Some(M2lRun::PlaneWave(None)),
                                "planewave-scan" => Some(M2lRun::PlaneWaveScan),
                                "planewave:1" => Some(M2lRun::PlaneWave(Some(Precision::Iprec1))),
                                "planewave:2" => Some(M2lRun::PlaneWave(Some(Precision::Iprec2))),
                                "planewave:3" => Some(M2lRun::PlaneWave(Some(Precision::Iprec3))),
                                _ => v
                                    .strip_prefix("auto:")?
                                    .parse()
                                    .ok()
                                    .map(|d| M2lRun::Fixed(HelmholtzM2l::Auto, None, Some(d))),
                            }
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|m| parsed.m2l = m)
                        .is_some(),
                    "--dense-budget" => value
                        .parse::<usize>()
                        .ok()
                        .and_then(|mb| mb.checked_mul(1 << 20))
                        .map(|bytes| parsed.dense_budget = bytes)
                        .is_some(),
                    "--plane-wave-budget" => match value.as_str() {
                        "whole" => Some(usize::MAX),
                        mb => mb
                            .parse::<usize>()
                            .ok()
                            .and_then(|mb| mb.checked_mul(1 << 20)),
                    }
                    .map(|bytes| parsed.plane_wave_budget = Some(bytes))
                    .is_some(),
                    "--rule" => match value.as_str() {
                        "calibrated" => Some(false),
                        "s7" => Some(true),
                        _ => None,
                    }
                    .map(|s7| parsed.s7 = s7)
                    .is_some(),
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
    if parsed.vectors == 0
        || parsed.threads.contains(&0)
        || parsed.kds.is_empty()
        || parsed.p2p.is_empty()
        || parsed.m2l.is_empty()
    {
        usage();
    }
    parsed
}

/// The M2L of a row: `rotation`, or `dense l…`, `classes l…`, `batched l…` or
/// `batched-fused l…` with the levels that ran the setting's tables and their MB; under
/// Auto prefixed with `auto ≤ d: ` (`∞` for no bound); `planewave l:iq…` with each level's
/// precision, its MB, and the waves buffer's MB, chunks, level calls and waves computed.
fn m2l_string(row: &Row) -> String {
    if row.m2l == HelmholtzM2l::PlaneWave {
        let levels: Vec<String> = (0..row.plane_wave.len())
            .filter_map(|l| Some(format!("{l}:i{}", row.plane_wave[l]?.iprec())))
            .collect();
        return match levels.is_empty() {
            true => "rotation".into(),
            false => format!(
                "planewave {} ({:.0} MB; waves {:.0} MB, {} chunks in {} calls, {} waves)",
                levels.join(","),
                row.dense_bytes as f64 / 1e6,
                row.plane_wave_buffer as f64 / 1e6,
                row.plane_wave_counts.chunks,
                row.plane_wave_counts.level_calls,
                row.plane_wave_counts.waves,
            ),
        };
    }
    let (m2l, mul_add, dense, bytes) = (
        row.m2l,
        row.m2l_mul_add,
        row.dense_levels.as_slice(),
        row.dense_bytes,
    );
    let levels: Vec<String> = (0..dense.len())
        .filter(|&l| dense[l])
        .map(|l| l.to_string())
        .collect();
    let name = match m2l {
        HelmholtzM2l::Classes => "classes",
        HelmholtzM2l::Batched | HelmholtzM2l::Auto if mul_add == MulAdd::Fused => "batched-fused",
        HelmholtzM2l::Batched | HelmholtzM2l::Auto => "batched",
        _ => "dense",
    };
    let ran = match levels.is_empty() {
        true => "rotation".into(),
        false => format!("{name} {} ({:.0} MB)", levels.join(","), bytes as f64 / 1e6),
    };
    match row.m2l_max_degree {
        Some(usize::MAX) => format!("auto ≤ ∞: {ran}"),
        Some(d) => format!("auto ≤ {d}: {ran}"),
        None => ran,
    }
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
            "| Workload | N | kD | degrees | charges | m | P2P | M2L | ranks × threads | levels \
             | leaf level | p by level | leaves | max leaf | build s | evaluate s | φ | ∇φ \
             | checksum | ÷ Rotation |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: \
             | ---: | ---: | ---: | ---: | --- | --- |"
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
            case.dense_budget = arguments.dense_budget;
            case.plane_wave_budget = arguments.plane_wave_budget;
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
                Degrees::Tolerance(eps) if arguments.s7 => format!("S7 ε = {eps:e}"),
                Degrees::Tolerance(eps) => format!("ε = {eps:e}"),
                Degrees::PerLevel(p) => format!("p = {}", p[0]),
            };
            if let (true, &Degrees::Tolerance(eps)) = (arguments.s7, degrees) {
                case.degrees = Degrees::PerLevel(vec![1; 64]);
                let probe = run(&case, &problem, &oracle, oracle_threads, &world)
                    .expect("the FMM at degree 1 runs");
                let k = Wavenumber::new(kd, arguments.kim).expect("a valid wavenumber");
                case.degrees = Degrees::PerLevel(
                    (0..64)
                        .map(|l| match l {
                            2.. if l < probe.levels => s7_degree(&k, probe.circumradii[l], eps),
                            _ => 0,
                        })
                        .collect(),
                );
            }
            let charges = match case.charges {
                ChargeKind::Real => "real",
                ChargeKind::Complex => "complex",
            };
            let print = |row: &Row, rotation: Option<[f64; 2]>| {
                let all = row.errors.last().expect("an entry over all vectors");
                let ratio = match rotation {
                    Some(r) if case.gradients => {
                        format!("{:.3}/{:.3}", all[0] / r[0], all[1] / r[1])
                    }
                    Some(r) => format!("{:.3}", all[0] / r[0]),
                    None => "–".into(),
                };
                let gradient = if case.gradients {
                    format!("{:.2e}", all[1])
                } else {
                    "–".into()
                };
                println!(
                    "| {} | {} | {kd_string} | {degrees_name} | {charges} | {} | {} | {} | {} × {} \
                     | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2e} | {gradient} \
                     | {:016x} | {ratio} |",
                    workload.name(),
                    problem.len(),
                    case.vectors,
                    row.p2p,
                    m2l_string(row),
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
            };
            // The errors of the case's Rotation rows by P2P kernel and threads.
            let mut rotation: Vec<((P2pChoice, usize), [f64; 2])> = Vec::new();
            let k = complex_k(&case);
            for &m2l_run in &arguments.m2l {
                for &p2p in &arguments.p2p {
                    for &threads in &arguments.threads {
                        case.p2p = p2p;
                        // The runs of the entry: (setting, mode, crossover, precisions),
                        // one, or the sweep's (the first ∞, the others from its degrees),
                        // or the scan's (Rotation first, the levels from its row).
                        let fixed = |m2l, mul_add, max_degree| {
                            (m2l, mul_add, max_degree, PlaneWavePrecision::Rule)
                        };
                        let mut runs = vec![match m2l_run {
                            M2lRun::Fixed(m2l, mul_add, max_degree) => {
                                fixed(m2l, mul_add, max_degree)
                            }
                            M2lRun::Sweep => fixed(HelmholtzM2l::Auto, None, Some(usize::MAX)),
                            M2lRun::PlaneWave(precision) => (
                                HelmholtzM2l::PlaneWave,
                                None,
                                None,
                                match precision {
                                    None => PlaneWavePrecision::Rule,
                                    Some(q) => PlaneWavePrecision::PerLevel(vec![Some(q); 64]),
                                },
                            ),
                            M2lRun::PlaneWaveScan => fixed(HelmholtzM2l::Rotation, None, None),
                        }];
                        let mut next = 0;
                        while next < runs.len() {
                            (
                                case.m2l,
                                case.m2l_mul_add,
                                case.m2l_max_degree,
                                case.plane_wave,
                            ) = runs[next].clone();
                            next += 1;
                            match run(&case, &problem, &oracle, threads, &world) {
                                Ok(row) => {
                                    let all =
                                        *row.errors.last().expect("an entry over all vectors");
                                    let reference = rotation
                                        .iter()
                                        .find(|(key, _)| *key == (p2p, threads))
                                        .map(|&(_, errors)| errors);
                                    let is_rotation = case.m2l == HelmholtzM2l::Rotation;
                                    if root {
                                        print(&row, reference.filter(|_| !is_rotation));
                                    }
                                    if is_rotation && reference.is_none() {
                                        rotation.push(((p2p, threads), all));
                                    }
                                    if next == 1 {
                                        match m2l_run {
                                            M2lRun::Sweep => runs.extend(
                                                sweep_crossovers(&row.degrees)[1..].iter().map(
                                                    |&d| fixed(HelmholtzM2l::Auto, None, Some(d)),
                                                ),
                                            ),
                                            M2lRun::PlaneWaveScan => {
                                                for l in plane_wave_levels(&row, k) {
                                                    for q in Precision::ALL {
                                                        let mut named = vec![None; row.levels];
                                                        named[l] = Some(q);
                                                        runs.push((
                                                            HelmholtzM2l::PlaneWave,
                                                            None,
                                                            None,
                                                            PlaneWavePrecision::PerLevel(named),
                                                        ));
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                Err(error) => {
                                    if root {
                                        eprintln!(
                                            "{} at kD = {kd_string}: {error}",
                                            workload.name()
                                        );
                                    }
                                }
                            }
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
    if root && let Some((bytes, how)) = resident_memory() {
        println!();
        println!("Rank 0: {:.0} MB {how}.", bytes / 1e6);
    }
}

//! The cost of dipole sources (Phase 6 T8, C6.4; docs/phase6/T8-dipoles.md,
//! "Measurements"): the evaluation of point dipoles, and of a charge and a dipole at every
//! point, against the evaluation of charges on the same tree, stage by stage, and on a
//! device the share of the host fallback of P2M, P2L and P2P in a dipole evaluation (the
//! 20 % rule of docs/design/optimisation.md §7.4). Prints Markdown on stdout and progress
//! on stderr.
//!
//! The problem: N points (`--n`, default 10⁶) uniform in [−1, 1)³ (`--dist cube`, the C3.2
//! distribution) or in the Plummer sphere of scale 0.1 (`--dist plummer`, C3.3), sources
//! equal to targets, the default tree (64 points per leaf), gradients on, charges uniform
//! in [−1, 1) and moments uniform in [−1, 1)³. For every degree of `--degrees` (default
//! 3, 8 and 12 in f64, 3 and 8 in f32) and source type the FMM is built once, evaluated once
//! as a warm-up, then `--repeats` times (default 5); the median wall time of the
//! evaluation and of its stages (`StageTimings`) are reported, with the ratio to charges.
//! With a device backend (`--backend`, the example built with its feature) one more
//! evaluation runs with `KindTiming::Synchronous` (a sync after every level call) and the
//! report gives the time of each kind there and the share of P2M, P2L and P2P, which run on
//! the host fallback with dipoles, in that evaluation.
//!
//! Run in release mode, on one rank, with every BLAS thread variable at 1 (outside the
//! macOS sandbox for Metal); MPI is initialised at `Threading::Funneled`, for `--threads`:
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example dipole_cost -- --n 1000000 --threads 1
//! cargo run --release -p nd-fmm-validate --features metal --example dipole_cost -- \
//!     --backend metal --precision f32
//! ```
//!
//! Timings are wall times on this machine, reported and never asserted.

use std::time::Duration;

use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{
    Backend, FmmBuilder, KindTiming, OperatorKind, SourceType, Sources, StageTimings,
};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_validate::bench::{cpu_model, target, toolchain};
use nd_fmm_validate::{SplitMix64, points};

/// The command line.
struct Arguments {
    n: usize,
    dist: String,
    degrees: Option<Vec<usize>>,
    precision: String,
    threads: usize,
    backend: Backend,
    repeats: usize,
}

/// Parses the flags of the module documentation; exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: dipole_cost [--n N] [--dist cube|plummer] [--degrees p,p,…] \
             [--precision f64|f32] [--threads t] [--backend host|cpu|metal|cuda] \
             [--repeats r]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        n: 1_000_000,
        dist: "cube".to_owned(),
        degrees: None,
        precision: "f64".to_owned(),
        threads: 1,
        backend: Backend::Host,
        repeats: 5,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match flag.as_str() {
            "--n" => parsed.n = value.parse().unwrap_or_else(|_| usage()),
            "--dist" if value == "cube" || value == "plummer" => parsed.dist = value.clone(),
            "--degrees" => {
                parsed.degrees = Some(
                    value
                        .split(',')
                        .map(|p| p.parse().unwrap_or_else(|_| usage()))
                        .collect(),
                )
            }
            "--precision" if value == "f64" || value == "f32" => parsed.precision = value.clone(),
            "--threads" => parsed.threads = value.parse().unwrap_or_else(|_| usage()),
            "--backend" => parsed.backend = value.parse().unwrap_or_else(|_| usage()),
            "--repeats" => parsed.repeats = value.parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() || parsed.repeats == 0 || parsed.threads == 0 || parsed.n < 1000 {
        usage();
    }
    parsed
}

/// The median of `times`, in milliseconds.
fn median_ms(mut times: Vec<Duration>) -> f64 {
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1e3
}

/// How a stage is read from `StageTimings`.
type Stage = fn(&StageTimings) -> Duration;

/// The stages of the report: their names and how to read each from `StageTimings`.
const STAGES: [(&str, Stage); 4] = [
    ("load", |t| t.forward_charges + t.load),
    ("upward local", |t| t.upward_local),
    ("downward", |t| t.downward),
    ("leaves", |t| t.evaluate_leaves),
];

/// One measured configuration.
struct Row {
    p: usize,
    kind: SourceType,
    total_ms: f64,
    stages_ms: [f64; 4],
    /// With a device: the share of P2M, P2L and P2P in the synchronous evaluation, and the
    /// time of each kind there.
    fallback: Option<(f64, Vec<(OperatorKind, f64)>)>,
}

/// Builds and times one source type at degree p in precision T.
fn measure<T: Stored + SimdScalar + Equivalence + Default>(
    args: &Arguments,
    p: usize,
    kind: SourceType,
    (points, charges, dipoles): (&[[f64; 3]], &[f64], &[[f64; 3]]),
    comm: &SimpleCommunicator,
) -> Row {
    let q: Vec<T> = charges.iter().map(|&c| T::from_f64(c)).collect();
    let d: Vec<[T; 3]> = dipoles.iter().map(|d| d.map(T::from_f64)).collect();
    let sources = match kind {
        SourceType::Charges => Sources::charges(&q),
        SourceType::Dipoles => Sources::dipoles(&d),
        _ => Sources::charges_and_dipoles(&q, &d),
    };
    let mut fmm = FmmBuilder::<T>::new(p)
        .gradients(true)
        .sources(kind)
        .threads(args.threads)
        .backend(args.backend)
        .build(points, points, comm)
        .unwrap_or_else(|error| panic!("p = {p}, {kind}: {error}"));
    fmm.evaluate_sources(&sources)
        .expect("the warm-up evaluates");
    let mut totals = Vec::new();
    let mut stages: [Vec<Duration>; 4] = Default::default();
    for _ in 0..args.repeats {
        let output = fmm.evaluate_sources(&sources).expect("the FMM evaluates");
        totals.push(output.timings.total());
        for (s, (_, read)) in stages.iter_mut().zip(STAGES) {
            s.push(read(&output.timings));
        }
    }
    let fallback = args.backend.is_device().then(|| {
        fmm.set_kind_timings(KindTiming::Synchronous)
            .expect("synchronous kind timings");
        let output = fmm.evaluate_sources(&sources).expect("the FMM evaluates");
        let kinds = output.timings.kinds.expect("kind timings");
        let total = output.timings.total().as_secs_f64();
        let per_kind: Vec<(OperatorKind, f64)> = kinds
            .iter()
            .filter(|(_, time)| time.calls > 0)
            .map(|(kind, time)| (kind, time.total().as_secs_f64() * 1e3))
            .collect();
        let host: f64 = per_kind
            .iter()
            .filter(|(k, _)| matches!(k, OperatorKind::P2m | OperatorKind::P2l | OperatorKind::P2p))
            .map(|(_, ms)| ms / 1e3)
            .sum();
        (host / total, per_kind)
    });
    let row = Row {
        p,
        kind,
        total_ms: median_ms(totals),
        stages_ms: stages.map(median_ms),
        fallback,
    };
    eprintln!("p = {p}, {kind}: {:.1} ms", row.total_ms);
    row
}

fn main() {
    let args = arguments();
    let (universe, _) =
        mpi::initialize_with_threading(mpi::Threading::Funneled).expect("MPI initialises");
    let comm = universe.world();
    if comm.size() != 1 {
        eprintln!("dipole_cost runs on one rank");
        std::process::exit(2);
    }
    let n = args.n;
    let mut rng = SplitMix64::new(0xd1b0_c057);
    let positions = if args.dist == "cube" {
        points::cube(&mut rng, n, [0.0; 3], 1.0)
    } else {
        points::plummer(&mut rng, n, [0.0; 3], 0.1)
    };
    let charges = points::charges(&mut rng, n);
    let dipoles: Vec<[f64; 3]> = (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
        .collect();
    let degrees = args.degrees.clone().unwrap_or_else(|| {
        if args.precision == "f64" {
            vec![3, 8, 12]
        } else {
            vec![3, 8]
        }
    });
    let mut rows = Vec::new();
    for &p in &degrees {
        for kind in [
            SourceType::Charges,
            SourceType::Dipoles,
            SourceType::ChargesAndDipoles,
        ] {
            let data = (&positions[..], &charges[..], &dipoles[..]);
            rows.push(if args.precision == "f64" {
                measure::<f64>(&args, p, kind, data, &comm)
            } else {
                measure::<f32>(&args, p, kind, data, &comm)
            });
        }
    }

    println!("# The cost of dipole sources (Phase 6 T8)\n");
    println!(
        "Machine: {}; {}; {}. Backend {}; {} thread(s); the {} at N = {n}, sources equal to \
         targets, the default tree (64 points per leaf), gradients on, {}; medians over {} \
         evaluations after a warm-up; wall times in ms.\n",
        cpu_model(),
        toolchain(),
        target(),
        args.backend,
        args.threads,
        args.dist,
        args.precision,
        args.repeats
    );
    println!("| p | sources | evaluation | / charges | load | upward local | downward | leaves |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for row in &rows {
        let base = rows
            .iter()
            .find(|r| r.p == row.p && r.kind == SourceType::Charges)
            .expect("the charge row");
        let stages: Vec<String> = row
            .stages_ms
            .iter()
            .zip(&base.stages_ms)
            .map(|(s, b)| format!("{s:.1} ({:.2})", s / b))
            .collect();
        println!(
            "| {} | {} | {:.1} | {:.2} | {} |",
            row.p,
            row.kind,
            row.total_ms,
            row.total_ms / base.total_ms,
            stages.join(" | ")
        );
    }
    println!("\nEach stage as ms (ratio to the charges' stage).");
    if args.backend.is_device() {
        println!(
            "\n## The host fallback on {} (one evaluation with synchronous kind timings)\n",
            args.backend
        );
        println!("| p | sources | P2M, P2L and P2P / evaluation | ms per kind |");
        println!("| --- | --- | --- | --- |");
        for row in &rows {
            let (share, kinds) = row.fallback.as_ref().expect("a device row");
            let kinds: Vec<String> = kinds
                .iter()
                .map(|(kind, ms)| format!("{kind} {ms:.2}"))
                .collect();
            println!(
                "| {} | {} | {:.1} % | {} |",
                row.p,
                row.kind,
                100.0 * share,
                kinds.join(", ")
            );
        }
    }
}

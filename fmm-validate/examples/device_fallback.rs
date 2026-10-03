//! The device path with every operator kind on the host fallback (Phase 4 T5, C4.1),
//! on the C3.2 problem: the transfers per evaluation by kind of data, and the
//! evaluation time against the host path, a baseline of the transfer cost. Prints
//! Markdown on stdout and progress on stderr.
//!
//! The problem is that of the C3.2 gate (`nd-fmm-exec`'s `tests/accuracy.rs`): N = 10⁵
//! points uniform in [−1, 1)³, sources equal to targets, a uniform level-4 tree
//! (`max_level` 4, one point per leaf as the refinement target), charges uniform in
//! [−1, 1), here with gradients, at p = 3 and 8, in f32 and, where the device supports
//! it, f64. Each configuration is built on the host and on each device backend at one
//! thread; one evaluation of each is a warm-up (it compiles the device kernels), then
//! the host and the device evaluate the same charges alternately, `--repeats` times
//! (default 5), and the median wall time of `Fmm::evaluate` is reported with the
//! number of runs. The device output equals the host output bit for bit in every run
//! (printed, and the example exits with an error otherwise).
//!
//! Run in release mode, on one rank, outside the macOS sandbox for Metal:
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --features cpu,metal --example device_fallback
//! cargo run --release -p nd-fmm-validate --features metal --example device_fallback -- \
//!     --backend metal --n 100000 --repeats 5
//! ```
//!
//! Without `--backend` every backend compiled in runs; a backend whose device cannot be
//! opened is listed as not run, with the reason. Timings are wall times on this
//! machine, reported and never asserted. MPI is initialised with `Threading::Single`.

use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::device::{DataKind, DeviceCounters, Traffic};
use nd_fmm_exec::fmm::{Backend, FmmBuilder, OperatorKind, Output};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cpu_model, target, toolchain};

/// The degrees.
const DEGREES: [usize; 2] = [3, 8];

/// The command line.
struct Arguments {
    backends: Vec<Backend>,
    n: usize,
    repeats: usize,
}

/// Parses `--backend b` (repeatable), `--n N` and `--repeats r`; exits with a message
/// on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: device_fallback [--backend cpu|metal|cuda]... [--n N] [--repeats r], \
             N >= 1000, r >= 1; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        backends: Vec::new(),
        n: 100_000,
        repeats: 5,
    };
    let mut rest = args.as_slice();
    while let [flag, value, tail @ ..] = rest {
        match (flag.as_str(), value.parse::<Backend>()) {
            ("--backend", Ok(backend)) if backend.is_device() => parsed.backends.push(backend),
            ("--n", _) => match value.parse() {
                Ok(n) if n >= 1000 => parsed.n = n,
                _ => usage(),
            },
            ("--repeats", _) => match value.parse() {
                Ok(r) if r >= 1 => parsed.repeats = r,
                _ => usage(),
            },
            _ => usage(),
        }
        rest = tail;
    }
    if !rest.is_empty() {
        usage();
    }
    if parsed.backends.is_empty() {
        parsed.backends = Backend::ALL
            .into_iter()
            .filter(|b| b.is_device() && b.is_compiled())
            .collect();
    }
    parsed
}

/// The output as bit patterns.
fn bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut values: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        values.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    values
}

/// The median of `times`, in milliseconds.
fn median_ms(times: &mut [Duration]) -> f64 {
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1e3
}

/// Bytes as MB (10⁶ bytes).
fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1e6
}

/// One measured configuration.
struct Row {
    backend: Backend,
    precision: &'static str,
    p: usize,
    host_ms: f64,
    device_ms: f64,
    counters: DeviceCounters,
    minimum: u64,
}

/// Builds and times one configuration in precision `T`; `None` if the device refuses
/// the precision.
fn measure<T: Stored + SimdScalar + mpi::traits::Equivalence + Default>(
    backend: Backend,
    p: usize,
    points: &[[f64; 3]],
    charges: &[f64],
    repeats: usize,
    comm: &SimpleCommunicator,
) -> Option<Row> {
    let charges: Vec<T> = charges.iter().map(|&q| T::from_f64(q)).collect();
    let builder = FmmBuilder::<T>::new(p)
        .max_level(4)
        .max_points_per_leaf(1)
        .gradients(true);
    let mut host = builder
        .build(points, points, comm)
        .expect("the host FMM builds");
    let start = Instant::now();
    // Every kind on the host fallback (from Phase 4 T6 P2P, and from T7 P2M, P2L, L2P and
    // M2P, run on the device by default).
    let mut device = match builder
        .clone()
        .backend(backend)
        .host_fallback(OperatorKind::ALL)
        .build(points, points, comm)
    {
        Ok(fmm) => fmm,
        Err(error) => {
            eprintln!("{backend}, {}: {error}", T::type_name());
            return None;
        }
    };
    let build = start.elapsed();
    let want = bits(&host.evaluate(&charges).expect("the host FMM evaluates"));
    let warm = bits(&device.evaluate(&charges).expect("the device FMM evaluates"));
    if warm != want {
        eprintln!("{backend}, p = {p}: the device output differs from the host output");
        std::process::exit(1);
    }
    let (mut host_times, mut device_times) = (Vec::new(), Vec::new());
    for _ in 0..repeats {
        let start = Instant::now();
        let output = host.evaluate(&charges).unwrap();
        host_times.push(start.elapsed());
        assert_eq!(bits(&output), want);
        let start = Instant::now();
        let output = device.evaluate(&charges).unwrap();
        device_times.push(start.elapsed());
        if bits(&output) != want {
            eprintln!("{backend}, p = {p}: the device output differs from the host output");
            std::process::exit(1);
        }
    }
    let counters = device.device_counters().expect("a device backend");
    let s = size_of::<T>() as u64;
    let minimum = (device.nsources() as u64 + 4 * device.ntargets() as u64) * s;
    eprintln!(
        "{backend} {} p = {p}: built in {:.2} s (device {:.2} s), bit for bit in {} runs",
        T::type_name(),
        build.as_secs_f64(),
        device.build_timings().device.as_secs_f64(),
        repeats + 1
    );
    Some(Row {
        backend,
        precision: T::type_name(),
        p,
        host_ms: median_ms(&mut host_times),
        device_ms: median_ms(&mut device_times),
        counters,
        minimum,
    })
}

/// The name of the float type, for reports.
trait TypeName {
    fn type_name() -> &'static str;
}

impl<T: RealScalar> TypeName for T {
    fn type_name() -> &'static str {
        if size_of::<T>() == 4 { "f32" } else { "f64" }
    }
}

/// "calls, MB" of one direction of `t`.
fn cell(calls: u64, bytes: u64) -> String {
    if calls == 0 {
        "–".to_owned()
    } else {
        format!("{calls}, {:.2} MB", mb(bytes))
    }
}

fn main() {
    let Arguments {
        backends,
        n,
        repeats,
    } = arguments();
    let universe = mpi::initialize().expect("MPI initialises");
    let comm = universe.world();
    if comm.size() != 1 {
        eprintln!("device_fallback runs on one rank: the device path needs one (until C5.1)");
        std::process::exit(2);
    }
    let mut rng = SplitMix64::new(0xc32);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
        .collect();
    let charges: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();

    println!("# The device path on the host fallback (Phase 4 T5)\n");
    println!(
        "Machine: {}; {}; {}. The C3.2 problem: N = {n} uniform in [−1, 1)³, sources equal \
         to targets, a uniform level-4 tree, gradients on, one thread. Every operator kind \
         runs on the host fallback; medians over {repeats} evaluations after a warm-up.\n",
        cpu_model(),
        toolchain(),
        target()
    );
    let mut rows = Vec::new();
    let mut not_run = Vec::new();
    for backend in backends {
        let info = match backend.probe() {
            Ok(info) => info,
            Err(reason) => {
                not_run.push(format!("{backend} ({reason})"));
                continue;
            }
        };
        println!("- {backend}: {info}");
        for p in DEGREES {
            rows.extend(measure::<f32>(
                backend, p, &points, &charges, repeats, &comm,
            ));
            if info.f64 {
                rows.extend(measure::<f64>(
                    backend, p, &points, &charges, repeats, &comm,
                ));
            }
        }
    }
    let ran: Vec<String> = {
        let mut names: Vec<String> = rows.iter().map(|r| r.backend.to_string()).collect();
        names.dedup();
        names
    };
    println!(
        "\nBackends run: {}; not run: {}.\n",
        if ran.is_empty() {
            "none".to_owned()
        } else {
            ran.join(", ")
        },
        if not_run.is_empty() {
            "none".to_owned()
        } else {
            not_run.join(", ")
        }
    );

    println!("## Transfers per evaluation, by kind of data (calls, MB)\n");
    println!(
        "| Backend | Precision | p | Charges up | Output down | Multipoles up / down | Locals up \
         / down | Target output up / down | Total up | Total down | Syncs | Launches | Minimum \
         (charges + output) |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for r in &rows {
        let t = |d: DataKind| -> Traffic { r.counters.evaluation_traffic.get(d) };
        let both = |d: DataKind| {
            format!(
                "{} / {}",
                cell(t(d).uploads, t(d).upload_bytes),
                cell(t(d).downloads, t(d).download_bytes)
            )
        };
        let total = r.counters.evaluation_traffic.total();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} MB |",
            r.backend,
            r.precision,
            r.p,
            cell(
                t(DataKind::Charges).uploads,
                t(DataKind::Charges).upload_bytes
            ),
            cell(
                t(DataKind::Output).downloads,
                t(DataKind::Output).download_bytes
            ),
            both(DataKind::FallbackMultipoles),
            both(DataKind::FallbackLocals),
            both(DataKind::FallbackTargetOutput),
            cell(total.uploads, total.upload_bytes),
            cell(total.downloads, total.download_bytes),
            r.counters.evaluation.syncs,
            r.counters.evaluation.launches,
            mb(r.minimum)
        );
    }

    println!("\n## Transfers per build (calls, MB uploaded)\n");
    println!("| Backend | Precision | p | Points | Indices | Geometry | Tables | Launches |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    for r in &rows {
        let t = |d: DataKind| {
            let t = r.counters.build_traffic.get(d);
            cell(t.uploads, t.upload_bytes)
        };
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.backend,
            r.precision,
            r.p,
            t(DataKind::Points),
            t(DataKind::Indices),
            t(DataKind::Geometry),
            t(DataKind::Tables),
            r.counters.build.launches
        );
    }

    println!("\n## Evaluation time, host path against the device path on the host fallback\n");
    println!("| Backend | Precision | p | Host (ms) | Device, full fallback (ms) | Ratio | Runs |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for r in &rows {
        println!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.2} | {repeats} |",
            r.backend,
            r.precision,
            r.p,
            r.host_ms,
            r.device_ms,
            r.device_ms / r.host_ms
        );
    }
}

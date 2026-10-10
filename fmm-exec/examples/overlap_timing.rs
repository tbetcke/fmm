//! The C5.2 measurement of Phase 5 T9 (docs/design/distributed-fmm.md §8.6, decision 11
//! of docs/phase5/README.md): the exposed waits of the overlapped exchanges against the
//! blocking exchange times, on any number of ranks. Release mode; nothing is asserted;
//! not registered for `run-examples` (a timing example).
//!
//! One build, evaluated with the exchanges blocking and overlapped in turn
//! (`Fmm::set_overlap`): one warm-up each, then `--evals` pairs. Per evaluation every
//! measure is the maximum over the ranks; the report gives the median over the
//! evaluations. The criterion: the median of max over ranks of exposed(sources) +
//! exposed(multipoles), against the median of max over ranks of `exchange_sources` +
//! `exchange_multipoles` in the blocking evaluations, at most 10%. The output of the two
//! paths is compared bit for bit on every evaluation (reported).
//!
//! The workloads are those of `nd-fmm-validate`'s `fmm_accuracy`: the cube [−1, 1)³, and
//! the Plummer sphere of scale 0.1 about the origin truncated at radius 1, drawn from seed
//! 0xc33 (SplitMix64), the charges uniform in [−1, 1) from seed 0xc34; every rank draws
//! all N and passes every P-th point (the `Fmm` moves them to their owners).
//!
//! On the M3 Max (`--mca` flags: root `CLAUDE.md`, "MPI"):
//!
//! ```sh
//! cargo build --release -p nd-fmm-exec --example overlap_timing
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 timeout 1800 \
//!   mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 8 \
//!   target/release/examples/overlap_timing --dist cube --n 1000000 --p 8
//! ```
//!
//! Options: `--dist cube|plummer`, `--n`, `--p`, `--precision f64|f32`, `--evals` (10),
//! `--threads` (1 per rank).

use std::time::{Duration, Instant};

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{Fmm, FmmBuilder, Output, StageTimings};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;

/// SplitMix64, as `nd-fmm-validate`'s generator.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

/// `n` points of `dist`, as `nd_fmm_validate::points::{cube, plummer}` draw them.
fn points(dist: &str, n: usize, rng: &mut SplitMix64) -> Vec<[f64; 3]> {
    match dist {
        "cube" => (0..n)
            .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
            .collect(),
        "plummer" => {
            let (scale, x) = (0.1, 10.0f64);
            let mass = x.powi(3) / (1.0 + x * x).powf(1.5);
            (0..n)
                .map(|_| {
                    let u = mass * rng.uniform();
                    let r = if u > 0.0 {
                        scale / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
                    } else {
                        0.0
                    };
                    let v = loop {
                        let v: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
                        let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        if (1e-6..=1.0).contains(&norm_sq) {
                            break v;
                        }
                    };
                    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    core::array::from_fn(|i| r * (v[i] / norm))
                })
                .collect()
        }
        other => panic!("unknown distribution {other}: cube or plummer"),
    }
}

/// The command line.
struct Args {
    dist: String,
    n: usize,
    p: usize,
    precision: String,
    evals: usize,
    threads: usize,
}

fn args() -> Args {
    let mut args = Args {
        dist: "cube".into(),
        n: 1_000_000,
        p: 8,
        precision: "f64".into(),
        evals: 10,
        threads: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        let number = || value.parse::<usize>().expect("a number");
        match flag.as_str() {
            "--dist" => args.dist = value.clone(),
            "--n" => args.n = number(),
            "--p" => args.p = number(),
            "--precision" => args.precision = value.clone(),
            "--evals" => args.evals = number(),
            "--threads" => args.threads = number(),
            other => panic!("unknown option {other}"),
        }
    }
    args
}

/// The measures of one evaluation on this rank, in seconds; see [`MEASURES`].
fn measures(t: &StageTimings, nlevels: usize) -> Vec<f64> {
    let s = |d: Duration| d.as_secs_f64();
    let o = t.overlap.unwrap_or_default();
    let mut m = vec![
        s(t.total()),
        s(t.exchange_sources),
        s(t.upward_local),
        s(t.upward_global),
        s(t.exchange_multipoles),
        s(t.downward),
        s(t.evaluate_leaves),
        s(t.exchange_sources + t.exchange_multipoles),
        s(o.sources.total),
        s(o.sources.exposed),
        s(o.sources.progress),
        s(o.multipoles.total),
        s(o.multipoles.exposed),
        s(o.multipoles.progress),
        s(o.sources.exposed + o.multipoles.exposed),
        s(o.coarse_gather),
        s(o.sources.sends),
        s(o.multipoles.sends),
        s(t.exchange_sources + t.upward_local + t.upward_global),
    ];
    m.extend(o.level_waits[..nlevels].iter().map(|&d| s(d)));
    m
}

/// The names of [`measures`], before the waits per level.
const MEASURES: [&str; 19] = [
    "evaluation (sum of the stages)",
    "exchange_sources",
    "upward_local",
    "upward_global",
    "exchange_multipoles",
    "downward",
    "evaluate_leaves",
    "exchange_sources + exchange_multipoles",
    "sources: total (post to completion)",
    "sources: exposed",
    "sources: progress (test calls)",
    "multipoles: total (post to completion)",
    "multipoles: exposed",
    "multipoles: progress (test calls)",
    "exposed(sources) + exposed(multipoles)",
    "coarse gather",
    "sources: exposed in the final wait for the sends",
    "multipoles: exposed in the final wait for the sends",
    "exchange_sources + upward_local + upward_global (to the end of the gather)",
];

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

fn bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut bits: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        bits.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    bits
}

/// The memory copy bandwidth of this rank while every rank copies, in bytes per second
/// (the median of 5 copies of 64 MB).
fn copy_bandwidth(comm: &SimpleCommunicator) -> f64 {
    let n = 8 << 20;
    let source = vec![1.0f64; n];
    let mut target = vec![0.0f64; n];
    let mut times = Vec::new();
    for _ in 0..5 {
        comm.barrier();
        let start = Instant::now();
        target.copy_from_slice(&source);
        times.push(start.elapsed().as_secs_f64());
        std::hint::black_box(&target);
    }
    (8 * n) as f64 / median(times)
}

fn run<T: Stored + SimdScalar + Equivalence + Default>(args: &Args, comm: &SimpleCommunicator) {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let all = points(&args.dist, args.n, &mut SplitMix64(0xc33));
    let mut rng = SplitMix64(0xc34);
    let all_charges: Vec<f64> = (0..args.n).map(|_| rng.range(-1.0, 1.0)).collect();
    let mine: Vec<usize> = (rank..args.n).step_by(size).collect();
    let sources: Vec<[f64; 3]> = mine.iter().map(|&i| all[i]).collect();
    let charges: Vec<T> = mine.iter().map(|&i| T::from_f64(all_charges[i])).collect();
    drop(all);

    let start = Instant::now();
    let mut fmm: Fmm<'_, T> = FmmBuilder::<T>::new(args.p)
        .gradients(true)
        .threads(args.threads)
        .build(&sources, &sources, comm)
        .unwrap_or_else(|error| panic!("rank {rank}: the FMM does not build: {error}"));
    let build = start.elapsed();
    let nlevels = fmm.nlevels();
    let bandwidth = copy_bandwidth(comm);

    // A warm-up of each path, then the pairs.
    let mut runs: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
    let mut differing = 0usize;
    for e in 0..=args.evals {
        let mut outputs = Vec::new();
        for (path, overlap) in [false, true].into_iter().enumerate() {
            fmm.set_overlap(overlap);
            comm.barrier();
            let output = fmm.evaluate(&charges).expect("the FMM evaluates");
            let local = measures(&output.timings, nlevels);
            let mut max = vec![0.0f64; local.len()];
            comm.all_reduce_into(&local[..], &mut max[..], SystemOperation::max());
            if e > 0 {
                runs[path].push(max);
            }
            outputs.push(bits(&output));
        }
        differing += outputs[0]
            .iter()
            .zip(&outputs[1])
            .filter(|(a, b)| a != b)
            .count();
    }
    let mut total_differing = 0usize;
    comm.all_reduce_into(&differing, &mut total_differing, SystemOperation::sum());

    // The traffic of every rank, gathered: per exchange the messages and values sent
    // and received.
    let traffic = fmm.exchange_traffic();
    let multipoles = traffic.multipoles.iter().fold([0usize; 4], |sum, t| {
        [
            sum[0] + t.messages_sent,
            sum[1] + t.messages_received,
            sum[2] + t.values_sent,
            sum[3] + t.values_received,
        ]
    });
    let local = [
        traffic.sources.messages_sent,
        traffic.sources.messages_received,
        traffic.sources.values_sent,
        traffic.sources.values_received,
        multipoles[0],
        multipoles[1],
        multipoles[2],
        multipoles[3],
        traffic.coarse_sent,
        traffic.coarse_received,
    ];
    let mut every = vec![0usize; local.len() * size];
    comm.all_gather_into(&local[..], &mut every[..]);
    let levels: Vec<usize> = traffic
        .multipoles
        .iter()
        .map(|t| t.values_received)
        .collect();
    let mut level_max = vec![0usize; levels.len()];
    comm.all_reduce_into(&levels[..], &mut level_max[..], SystemOperation::max());
    let mut bandwidths = vec![0.0f64; size];
    comm.all_gather_into(&bandwidth, &mut bandwidths[..]);

    if rank != 0 {
        return;
    }
    let value_bytes = size_of::<T>();
    let ms = |seconds: f64| seconds * 1e3;
    println!(
        "overlap_timing: {} N = {}, p = {}, {}, {size} ranks x {} threads, gradients, {} \
         evaluations per path after one warm-up; build {:.2} s; {} levels",
        args.dist,
        args.n,
        args.p,
        args.precision,
        args.threads,
        args.evals,
        build.as_secs_f64(),
        nlevels
    );
    println!(
        "BLAS variables: {}",
        [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"
        ]
        .map(|v| format!(
            "{v}={}",
            std::env::var(v).unwrap_or_else(|_| "unset".into())
        ))
        .join(" ")
    );
    println!(
        "output, overlapped against blocking: {total_differing} values differ over every \
         rank and evaluation"
    );
    println!();
    println!(
        "| measure, median over evaluations of the max over ranks | blocking, ms | overlapped, ms |"
    );
    println!("| --- | --- | --- |");
    let column = |path: usize, k: usize| median(runs[path].iter().map(|m| m[k]).collect());
    for (k, name) in MEASURES.iter().enumerate() {
        println!(
            "| {name} | {:.3} | {:.3} |",
            ms(column(0, k)),
            ms(column(1, k))
        );
    }
    for level in 0..nlevels {
        println!(
            "| multipoles: wait for level {level} | | {:.3} |",
            ms(column(1, MEASURES.len() + level))
        );
    }
    let blocking = column(0, 7);
    let exposed = column(1, 14);
    println!();
    println!(
        "C5.2 criterion: exposed {:.3} ms / blocking {:.3} ms = {:.1}% (at most 10%: {})",
        ms(exposed),
        ms(blocking),
        100.0 * exposed / blocking,
        if exposed <= 0.1 * blocking {
            "met"
        } else {
            "not met"
        }
    );
    println!(
        "evaluation: blocking {:.1} ms, overlapped {:.1} ms ({:+.2}%)",
        ms(column(0, 0)),
        ms(column(1, 0)),
        100.0 * (column(1, 0) / column(0, 0) - 1.0)
    );
    let slowest = bandwidths.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "copy bandwidth per rank, every rank copying: {:.1}-{:.1} GB/s",
        slowest / 1e9,
        bandwidths.iter().copied().fold(0.0, f64::max) / 1e9
    );
    println!();
    println!(
        "| rank | sources: msgs sent / received | sources: bytes sent / received | \
         multipoles: msgs sent / received | multipoles: bytes sent / received | coarse gather: \
         bytes sent / received | received bytes / copy bandwidth, ms |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for (r, t) in every.chunks_exact(local.len()).enumerate() {
        let received = (t[3] + t[7] + t[9]) * value_bytes;
        println!(
            "| {r} | {} / {} | {} / {} | {} / {} | {} / {} | {} / {} | {:.3} |",
            t[0],
            t[1],
            t[2] * value_bytes,
            t[3] * value_bytes,
            t[4],
            t[5],
            t[6] * value_bytes,
            t[7] * value_bytes,
            t[8] * value_bytes,
            t[9] * value_bytes,
            ms(received as f64 / bandwidths[r])
        );
    }
    println!();
    println!(
        "multipole bytes received per level, max over ranks: {:?}",
        level_max
            .iter()
            .map(|&v| v * value_bytes)
            .collect::<Vec<_>>()
    );
}

fn main() {
    let args = args();
    let threading = if args.threads > 1 {
        Threading::Funneled
    } else {
        Threading::Single
    };
    let (universe, _) = mpi::initialize_with_threading(threading).expect("MPI initialises once");
    let comm = universe.world();
    match args.precision.as_str() {
        "f64" => run::<f64>(&args, &comm),
        "f32" => run::<f32>(&args, &comm),
        other => panic!("unknown precision {other}: f64 or f32"),
    }
}

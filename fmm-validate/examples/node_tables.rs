//! The tables at scale and the node context (Phase 5N T3, docs/phase5n/T3-m2l-node-study.md):
//! what the M2L strategy's tables cost when every rank of a run needs them at once, and
//! what it costs a rank to learn its node. One configuration per launch, on any number of
//! ranks; Markdown on stdout from rank 0. Nothing is asserted; not registered for
//! `run-examples` (it times).
//!
//! In order, every rank at once, each step after a barrier and timed per rank (reported as
//! the maximum, minimum and mean over the ranks):
//! 1. **The node context**: `split_shared` of the world, then one all-reduce that agrees
//!    the fewest and the most ranks per node, as a strategy rule that sees the node would
//!    (requirement 5 of docs/phase5n/README.md). The first call, then the median of
//!    `--splits` more.
//! 2. **Every rank builds** (`Tables::build`), as `FmmBuilder::build` does without a
//!    table cache.
//! 3. **The first pass through the cache** (`Tables::load_or_build`) in
//!    `<dir>/node-tables-cache`: with `--fresh` rank 0 empties that directory first, so
//!    every rank builds and stores the same files at once (`TableCache::store`: a
//!    temporary file created with `create_new`, synced, renamed); otherwise it loads what
//!    an earlier launch stored. With `--mode node`, the lowest rank of each node goes
//!    first and the others wait at a barrier of the node, then load (one collective
//!    more); with `--mode every` (the default) every rank goes at once.
//! 4. **The check**: every rank loads again and compares with its tables of step 3, bit for
//!    bit; rank 0 lists the directory, temporary files left included.
//! 5. **Warm loads**: `--loads` more passes, every rank at once.
//!
//! The outcomes of steps 3 and 5 are counted over the ranks per table family (`Loaded`,
//! `Built`, `Rebuilt`, `NotStored`); a `NotStored` or `Rebuilt` is printed with its error
//! on stderr by the rank that saw it.
//!
//! Release mode, every BLAS thread variable at 1, under an external timeout. On the M3
//! Max (`--mca` flags: root `CLAUDE.md`, "MPI"):
//!
//! ```sh
//! cargo build --release -p nd-fmm-validate --example node_tables
//! timeout 600 mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 12 \
//!   target/release/examples/node_tables --p 8 --strategy dense --dir /tmp/fmm --fresh
//! ```
//!
//! Options:
//! - `--p p` (8), `--precision f64|f32` (f64), `--strategy dense|classes|rotation`
//!   (dense);
//! - `--dir DIR` the parent of the cache directory (required);
//! - `--fresh` empty the cache directory first;
//! - `--mode every|node` (every);
//! - `--loads k` warm passes (3), `--splits k` repeats of step 1 after the first (20).

use std::path::PathBuf;
use std::time::Instant;

use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_tables::cache::Stored;
use nd_fmm_tables::{CacheOutcome, TableCache};
use nd_fmm_validate::scaling::{SlurmJob, machine, median};

/// The command line.
struct Arguments {
    p: usize,
    precision: String,
    strategy: M2lStrategy,
    dir: PathBuf,
    fresh: bool,
    node_first: bool,
    loads: usize,
    splits: usize,
}

/// Parses the options of the module documentation; exits with a message on anything
/// else.
fn arguments() -> Arguments {
    let mut args = Arguments {
        p: 8,
        precision: "f64".into(),
        strategy: M2lStrategy::Dense,
        dir: PathBuf::new(),
        fresh: false,
        node_first: false,
        loads: 3,
        splits: 20,
    };
    let mut dir = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--fresh" {
            args.fresh = true;
            continue;
        }
        let value = it
            .next()
            .unwrap_or_else(|| fail(&format!("{flag} needs a value")));
        let number = || {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| fail(&format!("{flag}: {value} is not a number")))
        };
        match flag.as_str() {
            "--p" => args.p = number(),
            "--precision" => args.precision = value.clone(),
            "--strategy" => {
                args.strategy = match value.as_str() {
                    "dense" => M2lStrategy::Dense,
                    "classes" => M2lStrategy::Classes,
                    "rotation" => M2lStrategy::Rotation,
                    _ => fail(&format!(
                        "--strategy: dense, classes or rotation, not {value}"
                    )),
                };
            }
            "--dir" => dir = Some(PathBuf::from(&value)),
            "--mode" => {
                args.node_first = match value.as_str() {
                    "every" => false,
                    "node" => true,
                    _ => fail(&format!("--mode: every or node, not {value}")),
                };
            }
            "--loads" => args.loads = number(),
            "--splits" => args.splits = number(),
            other => fail(&format!("unknown option {other}")),
        }
    }
    if !["f64", "f32"].contains(&args.precision.as_str()) {
        fail(&format!("--precision: f64 or f32, not {}", args.precision));
    }
    args.dir = dir
        .unwrap_or_else(|| fail("--dir is required"))
        .join("node-tables-cache");
    args
}

fn fail(message: &str) -> ! {
    eprintln!("node_tables: {message}");
    std::process::exit(2);
}

/// The maximum, minimum and mean over the ranks of `seconds` (two all-reduces).
fn over_ranks(seconds: f64, comm: &SimpleCommunicator) -> [f64; 3] {
    let mut extremes = [0.0f64; 2];
    comm.all_reduce_into(
        &[seconds, -seconds][..],
        &mut extremes[..],
        SystemOperation::max(),
    );
    let mut sum = 0.0f64;
    comm.all_reduce_into(&seconds, &mut sum, SystemOperation::sum());
    [extremes[0], -extremes[1], sum / comm.size() as f64]
}

/// A table row of milliseconds: max, min, mean.
fn row(name: &str, [max, min, mean]: [f64; 3]) -> String {
    format!(
        "| {name} | {:.3} | {:.3} | {:.3} |",
        max * 1e3,
        min * 1e3,
        mean * 1e3
    )
}

/// The counts over the ranks of each outcome (Loaded, Built, Rebuilt, NotStored) per
/// family, as text; errors printed on stderr by the rank that saw them.
fn outcomes(
    pass: &str,
    local: &[(nd_fmm_tables::cache::TableKind, CacheOutcome)],
    comm: &SimpleCommunicator,
) -> String {
    let mut counts = vec![0u64; 4 * local.len()];
    for (f, (kind, outcome)) in local.iter().enumerate() {
        let k = match outcome {
            CacheOutcome::Loaded => 0,
            CacheOutcome::Built => 1,
            CacheOutcome::Rebuilt(error) => {
                eprintln!("rank {}: {pass}: {kind:?} rebuilt: {error}", comm.rank());
                2
            }
            CacheOutcome::NotStored(error) => {
                eprintln!("rank {}: {pass}: {kind:?} not stored: {error}", comm.rank());
                3
            }
        };
        counts[4 * f + k] = 1;
    }
    let mut total = vec![0u64; counts.len()];
    comm.all_reduce_into(&counts[..], &mut total[..], SystemOperation::sum());
    local
        .iter()
        .enumerate()
        .map(|(f, (kind, _))| {
            let c = &total[4 * f..4 * f + 4];
            format!(
                "{kind:?}: Loaded {}, Built {}, Rebuilt {}, NotStored {}",
                c[0], c[1], c[2], c[3]
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn run<T: Stored>(args: &Arguments, comm: &SimpleCommunicator) {
    let (rank, size) = (comm.rank(), comm.size());
    let mut lines = Vec::new();

    // 1. The node context: split_shared, then the agreement of the ranks per node.
    let (mut split, mut agree) = (Vec::new(), Vec::new());
    let mut per_node = [0i64; 2];
    let mut local_rank = 0;
    let mut node = None;
    for _ in 0..=args.splits {
        drop(node.take());
        comm.barrier();
        let start = Instant::now();
        let shared = comm.split_shared(rank);
        let split_time = start.elapsed().as_secs_f64();
        let n = i64::from(shared.size());
        comm.all_reduce_into(&[n, -n][..], &mut per_node[..], SystemOperation::max());
        split.push(split_time);
        agree.push(start.elapsed().as_secs_f64() - split_time);
        local_rank = shared.rank();
        node = Some(shared);
    }
    let node = node.expect("at least one split");
    let (fewest, most) = (-per_node[1], per_node[0]);
    let mut leaders = 0i32;
    comm.all_reduce_into(
        &i32::from(local_rank == 0),
        &mut leaders,
        SystemOperation::sum(),
    );
    let first_split = over_ranks(split[0], comm);
    let first_agree = over_ranks(agree[0], comm);
    let later_split = over_ranks(median(split[1..].to_vec()), comm);
    let later_agree = over_ranks(median(agree[1..].to_vec()), comm);
    lines.push(String::new());
    lines.push("### The node context: `split_shared` and the agreement".to_string());
    lines.push(String::new());
    lines.push(format!(
        "{leaders} node(s), {fewest}–{most} ranks per node. The first call, then the median \
         per rank of {} more; each after a barrier.",
        args.splits
    ));
    lines.push(String::new());
    lines.push("| step, ms | max | min | mean |".to_string());
    lines.push("| --- | --- | --- | --- |".to_string());
    lines.push(row("first `split_shared`", first_split));
    lines.push(row("first agreement (one all-reduce)", first_agree));
    lines.push(row("later `split_shared` (median)", later_split));
    lines.push(row("later agreement (median)", later_agree));

    // 2. Every rank builds.
    comm.barrier();
    let start = Instant::now();
    let built = Tables::<T>::build(args.p, args.strategy);
    let build = over_ranks(start.elapsed().as_secs_f64(), comm);
    drop(built);

    // 3. The first pass through the cache.
    if args.fresh && rank == 0 && args.dir.exists() {
        std::fs::remove_dir_all(&args.dir)
            .unwrap_or_else(|error| fail(&format!("cannot empty {}: {error}", args.dir.display())));
    }
    let cache = TableCache::new(&args.dir);
    comm.barrier();
    let start = Instant::now();
    let (first, first_outcomes) = if args.node_first {
        let mut first = None;
        if local_rank == 0 {
            first = Some(Tables::<T>::load_or_build(args.p, args.strategy, &cache));
        }
        node.barrier();
        first.unwrap_or_else(|| Tables::<T>::load_or_build(args.p, args.strategy, &cache))
    } else {
        Tables::<T>::load_or_build(args.p, args.strategy, &cache)
    };
    let first_time = over_ranks(start.elapsed().as_secs_f64(), comm);
    let first_counts = outcomes("first pass", &first_outcomes, comm);

    // 4. The check, bit for bit, and the directory.
    comm.barrier();
    let (again, _) = Tables::<T>::load_or_build(args.p, args.strategy, &cache);
    let mut differing = 0i32;
    comm.all_reduce_into(
        &i32::from(again != first),
        &mut differing,
        SystemOperation::sum(),
    );
    let listing = if rank == 0 {
        let mut entries: Vec<String> = std::fs::read_dir(&args.dir)
            .map(|dir| {
                dir.filter_map(Result::ok)
                    .map(|e| {
                        let bytes = e.metadata().map_or(0, |m| m.len());
                        format!("{} ({bytes} bytes)", e.file_name().to_string_lossy())
                    })
                    .collect()
            })
            .unwrap_or_default();
        entries.sort();
        entries.join(", ")
    } else {
        String::new()
    };

    // 5. Warm loads.
    let mut warm = Vec::new();
    let mut warm_outcomes = Vec::new();
    for _ in 0..args.loads {
        comm.barrier();
        let start = Instant::now();
        let (tables, outcomes) = Tables::<T>::load_or_build(args.p, args.strategy, &cache);
        warm.push(start.elapsed().as_secs_f64());
        drop(tables);
        warm_outcomes = outcomes;
    }
    let warm_time = (!warm.is_empty()).then(|| over_ranks(median(warm), comm));
    let warm_counts = (!warm_outcomes.is_empty()).then(|| outcomes("warm", &warm_outcomes, comm));

    if rank == 0 {
        println!(
            "## node_tables: {:?} at p = {}, {}, {size} ranks, {}",
            args.strategy,
            args.p,
            args.precision,
            if args.node_first {
                "one rank per node first"
            } else {
                "every rank at once"
            }
        );
        println!();
        println!("- machine: {}", machine());
        if let Some(job) = SlurmJob::from_env() {
            println!("- Slurm: {job}");
        }
        println!(
            "- cache directory: {} ({})",
            args.dir.display(),
            if args.fresh {
                "emptied first"
            } else {
                "as an earlier launch left it"
            }
        );
        for line in &lines {
            println!("{line}");
        }
        println!();
        println!("### The tables");
        println!();
        println!("| step, ms (per rank) | max | min | mean |");
        println!("| --- | --- | --- | --- |");
        println!("{}", row("every rank builds (`Tables::build`)", build));
        println!("{}", row("first pass through the cache", first_time));
        if let Some(t) = warm_time {
            println!(
                "{}",
                row(&format!("warm load (median of {})", args.loads), t)
            );
        }
        println!();
        println!("- first pass outcomes over the ranks: {first_counts}");
        if let Some(c) = &warm_counts {
            println!("- warm outcomes over the ranks: {c}");
        }
        println!(
            "- check: {differing} rank(s) loaded tables that differ from their first pass's \
             in any bit"
        );
        println!("- directory after the first pass: {listing}");
        println!();
        println!(
            "summary: strategy={:?} p={} precision={} ranks={size} nodes={leaders} \
             ranks_per_node={most} mode={} fresh={} split_first_ms={:.3} split_ms={:.3} \
             agree_ms={:.3} build_max_ms={:.3} first_max_ms={:.3} first_mean_ms={:.3} \
             warm_max_ms={:.3} differing={differing}",
            args.strategy,
            args.p,
            args.precision,
            if args.node_first { "node" } else { "every" },
            args.fresh,
            first_split[0] * 1e3,
            later_split[0] * 1e3,
            later_agree[0] * 1e3,
            build[0] * 1e3,
            first_time[0] * 1e3,
            first_time[2] * 1e3,
            warm_time.map_or(f64::NAN, |t| t[0] * 1e3),
        );
    }
}

fn main() {
    let args = arguments();
    let universe = mpi::initialize().expect("MPI initialises once");
    let comm = universe.world();
    match args.precision.as_str() {
        "f32" => run::<f32>(&args, &comm),
        _ => run::<f64>(&args, &comm),
    }
}

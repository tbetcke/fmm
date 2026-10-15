//! The replicated top's all-gather at R's sizes (Phase 5S T9, "The exposed collectives";
//! docs/phase5s/T9-scaling-at-scale.md): `MPI_Allgatherv` of a buffer of a given size in
//! all, split evenly over the ranks, timed on every rank, against the same gather in two
//! levels. One launch per placement; Markdown on stdout from rank 0. Nothing is asserted;
//! not registered for `run-examples` (it times).
//!
//! R holds at most `TOP_MAX_BOXES` = 4,096 boxes, so at the cap its multipoles are
//! 0.26 MB in all at f32 p = 3 (16 values of 4 B per box) and 2.65 MB at f64 p = 8 (81
//! values of 8 B), the default sizes. Per size and variant, after `--warmup` untimed
//! calls, `--reps` calls, each after a barrier and timed per rank; the report gives per
//! variant the median over the calls of the slowest rank's time (the call's wall time),
//! the fastest such call, the median over the calls of the mean over the ranks, and the
//! effective rate (the size over the median). The variants (`--variants`):
//! - **flat**: one `MPI_Allgatherv` on the world, as `TopExchange::gather` does. Open MPI
//!   picks the algorithm; `OMPI_MCA_coll_tuned_use_dynamic_rules=1` with
//!   `OMPI_MCA_coll_tuned_allgatherv_algorithm=k` forces one (Open MPI 4.1: 1 default
//!   (gatherv and broadcast), 2 Bruck, 3 ring, 4 neighbour exchange); the report prints
//!   every `OMPI_MCA_coll*` variable it sees.
//! - **two-level**: a gather-v on each node to its first rank (the node's leader), one
//!   all-gather-v of the nodes' blocks among the leaders, then a broadcast of the whole
//!   buffer on each node from its leader; the same buffer, in rank order, on every rank.
//!   Each part is also timed on its own (the slowest rank's median; a part's time on a
//!   rank includes its wait for the part before, so a non-leader's broadcast holds its
//!   leader's gathers). It needs every
//!   node's ranks to be consecutive (as `mpirun --map-by core` or `ppr:k:socket` places
//!   them); otherwise it is skipped and the report says so.
//! - **hierarchical**: an all-gather-v on each node, so that every rank holds its node's
//!   block; each node's block split into as many slices as the node has ranks, and the
//!   rank in place j of every node all-gathers the slices j of every node across the nodes
//!   (one communicator per place); then an all-gather-v on each node of what each rank
//!   received, copied into place. No broadcast; across the nodes each rank moves 1/k of
//!   the buffer. It also needs equal nodes; the parts are timed as above.
//!
//! The nodes are those of one `split_shared` of the world, or with `--node-size k` made-up
//! nodes of k consecutive ranks (to try the two-level gather on one machine). After the
//! timed calls every variant's buffer is checked value by value against the values every
//! rank sent; the report gives the values that differ, summed over the ranks.
//!
//! Release mode, under an external timeout. On the M3 Max (`--mca` flags: root
//! `CLAUDE.md`, "MPI"):
//!
//! ```sh
//! cargo build --release -p nd-fmm-validate --example collectives
//! timeout 300 mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 8 \
//!   target/release/examples/collectives --node-size 2
//! ```
//!
//! On Kathleen inside a Slurm allocation, for example 160 ranks on 4 nodes, Open MPI's
//! ring forced (tools/kathleen/README.md, "MPI and the launcher"):
//!
//! ```sh
//! OMPI_MCA_coll_tuned_use_dynamic_rules=1 OMPI_MCA_coll_tuned_allgatherv_algorithm=3 \
//!   timeout 600 mpirun -n 160 --map-by ppr:20:socket:PE=1 --bind-to core \
//!   target/release/examples/collectives
//! ```
//!
//! Options:
//! - `--bytes b[,b…]` the sizes in all, bytes (262144,2654208), each rounded down to
//!   whole f64 values;
//! - `--variants flat,two-level,hierarchical` (all three);
//! - `--reps k` timed calls per size and variant (20), `--warmup k` (3);
//! - `--node-size k` made-up nodes of k consecutive ranks (0: the nodes of
//!   `split_shared`).

use std::time::Instant;

use mpi::Count;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::{Communicator, CommunicatorCollectives, Group, Root};
use nd_fmm_validate::scaling::{SlurmJob, median};

/// The command line.
struct Arguments {
    bytes: Vec<usize>,
    flat: bool,
    two_level: bool,
    hierarchical: bool,
    reps: usize,
    warmup: usize,
    node_size: usize,
}

/// Parses the options of the module documentation; exits with a message on anything
/// else.
fn arguments() -> Arguments {
    let mut args = Arguments {
        bytes: vec![262_144, 2_654_208],
        flat: true,
        two_level: true,
        hierarchical: true,
        reps: 20,
        warmup: 3,
        node_size: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .unwrap_or_else(|| fail(&format!("{flag} needs a value")));
        let number = |v: &str| {
            v.parse::<usize>()
                .unwrap_or_else(|_| fail(&format!("{flag}: {v} is not a number")))
        };
        match flag.as_str() {
            "--bytes" => args.bytes = value.split(',').map(number).collect(),
            "--variants" => {
                args.flat = false;
                args.two_level = false;
                args.hierarchical = false;
                for variant in value.split(',') {
                    match variant {
                        "flat" => args.flat = true,
                        "two-level" => args.two_level = true,
                        "hierarchical" => args.hierarchical = true,
                        _ => fail(&format!(
                            "--variants: flat, two-level or hierarchical, not {variant}"
                        )),
                    }
                }
            }
            "--reps" => args.reps = number(&value).max(1),
            "--warmup" => args.warmup = number(&value),
            "--node-size" => args.node_size = number(&value),
            other => fail(&format!("unknown option {other}")),
        }
    }
    args
}

fn fail(message: &str) -> ! {
    eprintln!("collectives: {message}");
    std::process::exit(2);
}

/// The value rank `rank` sends at its position `i`: distinct per rank and position, exact
/// in f64.
fn value(rank: usize, i: usize) -> f64 {
    (rank * 1_000_003 + i) as f64
}

/// `total` values split evenly over `size` ranks, the remainder on the first ranks.
fn even_counts(total: usize, size: usize) -> Vec<Count> {
    (0..size)
        .map(|r| (total / size + usize::from(r < total % size)) as Count)
        .collect()
}

/// Exclusive prefix sums.
fn displacements(counts: &[Count]) -> Vec<Count> {
    counts
        .iter()
        .scan(0, |total, &count| {
            let start = *total;
            *total += count;
            Some(start)
        })
        .collect()
}

/// The node of this rank and the leaders' communicator (the nodes' first ranks; `None`
/// elsewhere), and whether every node's ranks are consecutive in the world.
struct Nodes {
    node: SimpleCommunicator,
    leaders: Option<SimpleCommunicator>,
    /// The ranks with this rank's place on their nodes, one per node, in node order.
    column: SimpleCommunicator,
    /// The world rank of the first rank of each node, in node order (every rank).
    first: Vec<usize>,
    /// The ranks of each node, in node order (every rank).
    sizes: Vec<usize>,
    consecutive: bool,
}

/// One split of `world` by node (`split_shared`, or made-up nodes of `node_size` ranks),
/// one by leaders, and one all-gather of each rank's node leader.
///
/// # Collective operation
///
/// On every rank of `world`.
fn nodes(world: &SimpleCommunicator, node_size: usize) -> Nodes {
    let rank = world.rank();
    let node = if node_size == 0 {
        world.split_shared(rank)
    } else {
        world
            .split_by_color_with_key(
                Color::with_value(rank / node_size as i32),
                mpi::topology::Key::from(rank),
            )
            .expect("every rank has a colour")
    };
    let leader = node
        .group()
        .translate_rank(0, &world.group())
        .expect("the node's ranks are ranks of the world") as u64;
    let leaders = world.split_by_color(if node.rank() == 0 {
        Color::with_value(0)
    } else {
        Color::undefined()
    });
    let column = world
        .split_by_color_with_key(
            Color::with_value(node.rank()),
            mpi::topology::Key::from(rank),
        )
        .expect("every rank has a colour");
    let mut leader_of = vec![0u64; world.size() as usize];
    world.all_gather_into(&leader, &mut leader_of[..]);
    let mut first: Vec<u64> = leader_of.clone();
    first.sort_unstable();
    first.dedup();
    let sizes: Vec<usize> = first
        .iter()
        .map(|f| leader_of.iter().filter(|&l| l == f).count())
        .collect();
    // Consecutive: the ranks of node k are first[k] .. first[k] + sizes[k].
    let consecutive = leader_of.iter().enumerate().all(|(r, &l)| {
        let k = first.binary_search(&l).expect("every leader is listed");
        r >= l as usize && r < l as usize + sizes[k]
    });
    Nodes {
        node,
        leaders,
        column,
        first: first.into_iter().map(|f| f as usize).collect(),
        sizes,
        consecutive,
    }
}

/// The per-call times of one rank, reduced: per call the slowest rank and the mean over
/// the ranks, then over the calls the median of each and the fastest slowest rank.
fn reduce(times: &[f64], comm: &SimpleCommunicator) -> [f64; 3] {
    let mut max = vec![0.0f64; times.len()];
    comm.all_reduce_into(times, &mut max[..], SystemOperation::max());
    let mut sum = vec![0.0f64; times.len()];
    comm.all_reduce_into(times, &mut sum[..], SystemOperation::sum());
    let size = comm.size() as f64;
    let fastest = max.iter().copied().fold(f64::INFINITY, f64::min);
    [
        median(max),
        fastest,
        median(sum.iter().map(|s| s / size).collect()),
    ]
}

/// The values of `gathered` that differ from what the ranks sent, summed over the ranks.
fn differing(gathered: &[f64], counts: &[Count], comm: &SimpleCommunicator) -> u64 {
    let mut wrong = 0u64;
    let mut at = 0;
    for (r, &count) in counts.iter().enumerate() {
        for i in 0..count as usize {
            if gathered[at].to_bits() != value(r, i).to_bits() {
                wrong += 1;
            }
            at += 1;
        }
    }
    let mut total = 0u64;
    comm.all_reduce_into(&wrong, &mut total, SystemOperation::sum());
    total
}

/// One row of the report: the variant, the median, the fastest, the mean, the rate.
fn row(name: &str, [median, fastest, mean]: [f64; 3], bytes: usize) -> String {
    format!(
        "| {name} | {:.3} | {:.3} | {:.3} | {:.2} |",
        median * 1e3,
        fastest * 1e3,
        mean * 1e3,
        bytes as f64 / median / 1e9
    )
}

fn main() {
    let args = arguments();
    let universe = mpi::initialize().expect("MPI initialises");
    let world = universe.world();
    let (rank, size) = (world.rank() as usize, world.size() as usize);
    let nodes = nodes(&world, args.node_size);
    let mut lines = Vec::new();

    if rank == 0 {
        let library = mpi::environment::library_version()
            .ok()
            .and_then(|v| {
                v.trim_matches(char::from(0))
                    .lines()
                    .next()
                    .map(|l| l.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string());
        let (fewest, most) = (
            nodes.sizes.iter().min().copied().unwrap_or(0),
            nodes.sizes.iter().max().copied().unwrap_or(0),
        );
        lines.push("## The replicated top's all-gather, micro-benchmarked".to_string());
        lines.push(String::new());
        lines.push(format!(
            "- {size} ranks on {} node(s) ({}), {fewest}–{most} ranks per node, \
             {}consecutive",
            nodes.sizes.len(),
            if args.node_size == 0 {
                "`split_shared`".to_string()
            } else {
                format!("made up, {} ranks each", args.node_size)
            },
            if nodes.consecutive { "" } else { "not " }
        ));
        lines.push(format!("- MPI library: {library}"));
        if let Some(job) = SlurmJob::from_env() {
            lines.push(format!(
                "- Slurm job {}, QoS {}, nodes {}",
                job.id,
                job.qos.as_deref().unwrap_or("–"),
                job.nodelist.as_deref().unwrap_or("–")
            ));
        }
        let mut mca: Vec<String> = std::env::vars()
            .filter(|(k, _)| k.starts_with("OMPI_MCA_coll"))
            .map(|(k, v)| format!("`{k}={v}`"))
            .collect();
        mca.sort();
        lines.push(format!(
            "- Open MPI's collective variables: {}",
            if mca.is_empty() {
                "none (the defaults)".to_string()
            } else {
                mca.join(", ")
            }
        ));
        lines.push(format!(
            "- {} timed calls per size and variant after {} untimed, each after a barrier",
            args.reps, args.warmup
        ));
        lines.push(String::new());
        lines.push(
            "| size, variant | median of the slowest rank (ms) | fastest call (ms) | median \
             of the mean (ms) | rate (GB/s) |"
                .to_string(),
        );
        lines.push("| --- | ---: | ---: | ---: | ---: |".to_string());
    }

    let mut checks = Vec::new();
    for &bytes in &args.bytes {
        let total = bytes / size_of::<f64>();
        let bytes = total * size_of::<f64>();
        let counts = even_counts(total, size);
        let displs = displacements(&counts);
        let mine: Vec<f64> = (0..counts[rank] as usize).map(|i| value(rank, i)).collect();
        let mut gathered = vec![0.0f64; total];
        let label = format!("{:.2} MB", bytes as f64 / 1e6);

        if args.flat {
            let mut times = Vec::with_capacity(args.reps);
            for call in 0..args.warmup + args.reps {
                gathered.fill(0.0);
                world.barrier();
                let start = Instant::now();
                world.all_gather_varcount_into(
                    &mine[..],
                    &mut PartitionMut::new(&mut gathered[..], &counts[..], &displs[..]),
                );
                if call >= args.warmup {
                    times.push(start.elapsed().as_secs_f64());
                }
            }
            let reduced = reduce(&times, &world);
            checks.push((
                format!("{label}, flat"),
                differing(&gathered, &counts, &world),
            ));
            if rank == 0 {
                lines.push(row(&format!("{label}, flat"), reduced, bytes));
            }
        }

        if args.two_level && nodes.consecutive {
            // The node's block: its ranks' values, contiguous in the world's order.
            let k = nodes
                .first
                .iter()
                .rposition(|&f| f <= rank)
                .expect("a node holds this rank");
            let (first, members) = (nodes.first[k], nodes.sizes[k]);
            let node_counts: Vec<Count> = counts[first..first + members].to_vec();
            let node_displs = displacements(&node_counts);
            let node_total: usize = node_counts.iter().map(|&c| c as usize).sum();
            let block_counts: Vec<Count> = nodes
                .first
                .iter()
                .zip(&nodes.sizes)
                .map(|(&f, &n)| counts[f..f + n].iter().sum())
                .collect();
            let block_displs: Vec<Count> = nodes.first.iter().map(|&f| displs[f]).collect();
            let mut block = vec![0.0f64; node_total];
            let mut parts: [Vec<f64>; 4] = Default::default();
            for call in 0..args.warmup + args.reps {
                gathered.fill(0.0);
                world.barrier();
                let start = Instant::now();
                let root = nodes.node.process_at_rank(0);
                if nodes.node.rank() == 0 {
                    root.gather_varcount_into_root(
                        &mine[..],
                        &mut PartitionMut::new(&mut block[..], &node_counts[..], &node_displs[..]),
                    );
                } else {
                    root.gather_varcount_into(&mine[..]);
                }
                let gathered_on_node = Instant::now();
                if let Some(leaders) = &nodes.leaders {
                    leaders.all_gather_varcount_into(
                        &block[..],
                        &mut PartitionMut::new(
                            &mut gathered[..],
                            &block_counts[..],
                            &block_displs[..],
                        ),
                    );
                }
                let gathered_by_leaders = Instant::now();
                root.broadcast_into(&mut gathered[..]);
                let end = Instant::now();
                if call >= args.warmup {
                    parts[0].push((end - start).as_secs_f64());
                    parts[1].push((gathered_on_node - start).as_secs_f64());
                    parts[2].push((gathered_by_leaders - gathered_on_node).as_secs_f64());
                    parts[3].push((end - gathered_by_leaders).as_secs_f64());
                }
            }
            let reduced: Vec<[f64; 3]> = parts.iter().map(|t| reduce(t, &world)).collect();
            checks.push((
                format!("{label}, two-level"),
                differing(&gathered, &counts, &world),
            ));
            if rank == 0 {
                lines.push(row(&format!("{label}, two-level"), reduced[0], bytes));
                lines.push(row(
                    &format!("{label}, part: gather-v on the node"),
                    reduced[1],
                    bytes,
                ));
                lines.push(row(
                    &format!("{label}, part: the leaders' all-gather-v"),
                    reduced[2],
                    bytes,
                ));
                lines.push(row(
                    &format!("{label}, part: broadcast on the node"),
                    reduced[3],
                    bytes,
                ));
            }
        }

        let equal = nodes.sizes.iter().all(|&n| n == nodes.sizes[0]);
        if args.hierarchical && nodes.consecutive && equal {
            // Node n's block, split into k slices by value; slice j goes across the nodes
            // from the rank in place j of each node.
            let k = nodes.sizes[0];
            let j = nodes.node.rank() as usize;
            let n_of = nodes
                .first
                .iter()
                .rposition(|&f| f <= rank)
                .expect("a node holds this rank");
            let node_counts: Vec<Count> = counts[nodes.first[n_of]..nodes.first[n_of] + k].to_vec();
            let node_displs = displacements(&node_counts);
            let block_total: Vec<usize> = nodes
                .first
                .iter()
                .map(|&f| counts[f..f + k].iter().map(|&c| c as usize).sum())
                .collect();
            let block_start: Vec<usize> = nodes.first.iter().map(|&f| displs[f] as usize).collect();
            let slice = |n: usize, j: usize| {
                let t = block_total[n];
                block_start[n] + t * j / k..block_start[n] + t * (j + 1) / k
            };
            let column_counts: Vec<Count> = (0..nodes.first.len())
                .map(|n| slice(n, j).len() as Count)
                .collect();
            let column_displs: Vec<Count> = (0..nodes.first.len())
                .map(|n| slice(n, j).start as Count)
                .collect();
            let own = slice(n_of, j);
            let mut send = vec![0.0f64; own.len()];
            let pack_counts: Vec<Count> = (0..k)
                .map(|jj| {
                    (0..nodes.first.len())
                        .map(|n| slice(n, jj).len() as Count)
                        .sum()
                })
                .collect();
            let pack_displs = displacements(&pack_counts);
            let mut pack = vec![0.0f64; pack_counts[j] as usize];
            let mut staging = vec![0.0f64; total];
            let mut parts: [Vec<f64>; 4] = Default::default();
            for call in 0..args.warmup + args.reps {
                gathered.fill(0.0);
                world.barrier();
                let start = Instant::now();
                let range = block_start[n_of]..block_start[n_of] + block_total[n_of];
                nodes.node.all_gather_varcount_into(
                    &mine[..],
                    &mut PartitionMut::new(
                        &mut gathered[range],
                        &node_counts[..],
                        &node_displs[..],
                    ),
                );
                let on_node = Instant::now();
                send.copy_from_slice(&gathered[own.clone()]);
                nodes.column.all_gather_varcount_into(
                    &send[..],
                    &mut PartitionMut::new(
                        &mut gathered[..],
                        &column_counts[..],
                        &column_displs[..],
                    ),
                );
                let across = Instant::now();
                let mut at = 0;
                for n in 0..nodes.first.len() {
                    let s = slice(n, j);
                    pack[at..at + s.len()].copy_from_slice(&gathered[s.clone()]);
                    at += s.len();
                }
                nodes.node.all_gather_varcount_into(
                    &pack[..],
                    &mut PartitionMut::new(&mut staging[..], &pack_counts[..], &pack_displs[..]),
                );
                for (jj, &displ) in pack_displs.iter().enumerate() {
                    let mut at = displ as usize;
                    for n in 0..nodes.first.len() {
                        let s = slice(n, jj);
                        if n != n_of {
                            gathered[s.clone()].copy_from_slice(&staging[at..at + s.len()]);
                        }
                        at += s.len();
                    }
                }
                let end = Instant::now();
                if call >= args.warmup {
                    parts[0].push((end - start).as_secs_f64());
                    parts[1].push((on_node - start).as_secs_f64());
                    parts[2].push((across - on_node).as_secs_f64());
                    parts[3].push((end - across).as_secs_f64());
                }
            }
            let reduced: Vec<[f64; 3]> = parts.iter().map(|t| reduce(t, &world)).collect();
            checks.push((
                format!("{label}, hierarchical"),
                differing(&gathered, &counts, &world),
            ));
            if rank == 0 {
                lines.push(row(&format!("{label}, hierarchical"), reduced[0], bytes));
                lines.push(row(
                    &format!("{label}, part: all-gather-v on the node"),
                    reduced[1],
                    bytes,
                ));
                lines.push(row(
                    &format!("{label}, part: the slices' all-gather-v across the nodes"),
                    reduced[2],
                    bytes,
                ));
                lines.push(row(
                    &format!("{label}, part: the slices' all-gather-v on the node, copies"),
                    reduced[3],
                    bytes,
                ));
            }
        }
    }

    if rank == 0 {
        lines.push(String::new());
        if args.two_level && !nodes.consecutive {
            lines.push(
                "The two-level gather was skipped: the nodes' ranks are not consecutive."
                    .to_string(),
            );
        }
        if args.hierarchical
            && !(nodes.consecutive && nodes.sizes.iter().all(|&n| n == nodes.sizes[0]))
        {
            lines.push(
                "The hierarchical gather was skipped: the nodes' ranks are not consecutive or \
                 the nodes differ in size."
                    .to_string(),
            );
        }
        let checked: Vec<String> = checks
            .iter()
            .map(|(name, wrong)| format!("{name}: {wrong}"))
            .collect();
        lines.push(format!(
            "Values that differ from those sent, summed over the ranks: {}.",
            checked.join("; ")
        ));
        println!("{}", lines.join("\n"));
    }
}

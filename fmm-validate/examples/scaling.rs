//! Strong and weak scaling of the distributed host FMM (Phase 5 T10, C5.3): one
//! configuration per launch, on any number of ranks; Markdown on stdout from rank 0.
//!
//! The core is `nd_fmm_validate::scaling`: every rank generates its points by index, the
//! FMM is built (the build by part, the median over `--builds`), evaluated `--evals`
//! times per path after one warm-up, and reported per rank and reduced over the ranks:
//! the work as built, the build, the evaluation stage by stage with the load imbalance of
//! the compute stages, the messages and bytes per exchange, the memory per rank (the
//! formula of docs/design/distributed-fmm.md §9.2, a *model*, and the resident size where
//! it can be read), and optionally the errors against the direct sum and the difference
//! from the one-rank FMM. Each path ends with a `summary:` line of `key=value` fields for
//! the sweeps' tables. Nothing is asserted; not registered for `run-examples` (it times).
//!
//! Release mode, every BLAS thread variable at 1, under an external timeout. On the M3
//! Max (`--mca` flags: root `CLAUDE.md`, "MPI"):
//!
//! ```sh
//! cargo build --release -p nd-fmm-validate --example scaling
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!   VECLIB_MAXIMUM_THREADS=1 timeout 1800 \
//!   mpirun --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 8 \
//!   target/release/examples/scaling --dist cube --n 1000000 --precision f64 --p 8 \
//!   --overlap both
//! ```
//!
//! On locust the same without the `--mca` flags, with `--report-bindings` (tools/gh200/
//! README.md, "MPI at n ranks"). `tools/scaling/run.sh` runs the sweeps of the T10 brief.
//! On a cluster (docs/design/distributed-fmm.md §11), for example 2 Kathleen nodes × 40
//! ranks inside a Slurm allocation (Phase 5N T2; `tools/kathleen/jobs/sweep.sbatch` runs
//! the sweeps):
//!
//! ```sh
//! mpirun -n 80 --map-by ppr:20:socket:PE=1 --bind-to core --report-bindings \
//!   target/release/examples/scaling --dist cube --n 10000000 --precision f64 --p 8 \
//!   --overlap both
//! ```
//!
//! Every report names where its ranks ran (`scaling::Placement`): the nodes and the ranks
//! per node from one `split_shared`, each rank's host and CPUs, the MPI library, and under
//! Slurm the job id, QoS, node list and the binding Slurm reports.
//!
//! **Rank-count invariance** (Phase 5S T2, C5S.1; `scaling::invariance`): with
//! `--invariance k[,k…]` the launch runs no timed evaluations; it runs the workload on the
//! P ranks of the launch and on P/k ranks for every k (the small runs on a split of the
//! world, rank r holding the inputs of ranks k r … k r + k − 1 concatenated), compares the
//! outputs bit for bit by index without gathering them, prints the values that differ per
//! check (the sum, the most on a rank, the ranks with any), and exits with status 1 on
//! every rank if any value differs. For example 512 against 64 ranks, oversubscribed on
//! locust (correctness, never a timing):
//!
//! ```sh
//! ulimit -n "$(ulimit -Hn)"
//! timeout 3600 mpirun --map-by :OVERSUBSCRIBE --bind-to none -n 512 \
//!   target/release/examples/scaling --dist cube --n 10000000 --precision f64 --p 8 \
//!   --strategy rotation --invariance 8
//! ```
//!
//! On Kathleen, `tools/kathleen/jobs/correctness.sbatch` runs it (80 against 40, 20 and 10
//! ranks at 2 nodes; 160 against 20 and 640 against 80 at 4).
//!
//! Options:
//! - `--dist cube|sphere|plummer|clusters` (cube);
//! - `--n N` (10⁶), or `--n-per-rank n` for N = n P (weak scaling);
//! - `--precision f64|f32` (f64), `--p p` (8);
//! - `--threads t` rayon threads per rank (1; MPI at `Threading::Funneled` above 1);
//! - `--strategy auto|dense|classes|rotation|batched|planewave`, the M2L strategy (auto;
//!   `batched` is the host's batched M2L of Phase 6 T2, `planewave` its plane-wave M2L of
//!   Phase 6 T13);
//! - `--p2p auto|reference|scalar|neon|avx2|avx512` the P2P kernel (`FmmBuilder::p2p_kernel`,
//!   auto; an ISA this machine lacks is refused before MPI starts; Phase 6 T6), which on
//!   the host also runs the batched M2L's kernel (the summary's `p2p`, resolved);
//! - `--mul-add auto|separate|fused` the batched M2L's multiply–add mode
//!   (`FmmBuilder::m2l_mul_add`; `auto`, the library's default of p since Phase 6 T11:
//!   fused from p = 6, else separate, bit for bit `dense`);
//! - `--budget MB` the node budget in MB of 10⁶ bytes (`FmmBuilder::dense_node_budget`;
//!   the library's 64 if not given; Phase 5N T4: `tools/scaling/run.sh` passes 56 on the
//!   M3 Max and 96 on locust and Kathleen; reported only since Phase 6 T11, whose rule of
//!   `auto` does not read it);
//! - `--overlap off|on|both` (both: one build, the evaluations alternating, a block each);
//! - `--input share|rank0|owners|ranges` (share; `ranges`: contiguous index ranges, found
//!   in O(N/P) time, for N above 10⁹);
//! - `--evals k` timed evaluations per path (10), `--builds b` (1);
//! - `--leaf-size n` the refinement target, points per leaf (`FmmBuilder::max_points_per_leaf`;
//!   Phase 6 T11; the library's default of p and the backend if not given, the summary's
//!   `leaf`);
//! - `--errors k` charge vectors of the errors against the direct sum at `--sampled s`
//!   targets (0, not checked; 1,000 targets);
//! - `--reference` the one-rank FMM over every point on rank 0, of the strategy the run
//!   resolved, against the output; refused (every rank exits with status 2 and the
//!   reason) above N = 2 × 10⁷ (`scaling::REFERENCE_MAX_POINTS`, Phase 5S T2), where the
//!   errors against the direct sum and `--invariance` are the oracles;
//! - `--reference-strategy auto|dense|classes|rotation|batched|planewave` that FMM with
//!   another M2L strategy (the difference between strategies; Phase 5N T3);
//! - `--kinds` every level call timed by operator kind (`KindTiming::Synchronous`): the
//!   M2M, M2L and L2L parts per rank and the time per V pair (Phase 5N T3);
//! - `--cut work|points` the blocks cut again by the modelled cost of their work (`work`,
//!   the library's default, `FmmBuilder::work_cut`; Phase 5S T7) or by points only;
//! - `--near off|cost|W,X` the near rule (`FmmBuilder::near_rule`, Phase 6 T12; `cost`,
//!   the library's default since Phase 6 T11): W and X pairs between leaves whose fine leaf holds fewer points
//!   than the crossovers run as P2P, with the crossovers of the cost model (`cost`) or
//!   named (`12,6`); the report gives the crossovers and the pairs converted;
//! - `--routing direct|tree|treeF` how the multipole exchange routes the multipoles
//!   (`FmmBuilder::multipole_routing`; Phase 5S T8s): along the forwarding tree with the
//!   library's fan-out (`tree`, the default) or with fan-out F (`tree4`), or from each
//!   owner to every reader (`direct`);
//! - `--alpha a,a,a,a,a,a,a,a` the eight constants of the modelled cost in ns per counted
//!   unit, P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P (`FmmBuilder::cost_model`; the library's
//!   `CostModel::calibrated`, measured on Kathleen, if not given); with `--kinds` the
//!   report fits them to the measured kind times (`scaling::ModelFit`, the `alpha_ns`
//!   field of the summary), which is how a machine's constants are calibrated;
//! - `--backend host|cpu|metal|cuda` where the operators run (`FmmBuilder::backend`; host;
//!   a device needs the example built with that feature of `nd-fmm-validate`): one rank per
//!   device for a timing (ranks sharing a device are a correctness check only);
//! - `--rhs m` charge vectors per evaluation (`FmmBuilder::vectors`, Phase 6 T5; 1): the
//!   FMM built for m vectors and timed through `Fmm::evaluate_many` (the summary's `rhs`
//!   field; the time per vector is the wall time over m); not with `--errors`,
//!   `--reference` or `--invariance`;
//! - `--device-scratch MB` the device's chunk budget (`FmmBuilder::device_scratch_budget`;
//!   the library's, per charge vector, if not given), `--device-gemm
//!   auto|library|hand|shared`, `--device-gather auto|separate|fused` and
//!   `--device-reduction by-chunk|walk` (`FmmBuilder::{device_gemm, device_gather,
//!   device_reduction}`; Phase 6 T7): the summary's `device_*` fields, with the scratch the
//!   device allocated (`device_scratch_bytes`);
//! - `--invariance k[,k…]` rank-count invariance against P/k ranks for each k (above),
//!   with the overlap of `--overlap` (the large run evaluates its paths, the small runs
//!   overlapped unless `--overlap off`) and the strategy of `--strategy` (`auto`: the one
//!   the large run resolves, named in the small runs).

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::Communicator;
use nd_fmm_exec::cost::CostModel;
use nd_fmm_exec::fmm::{
    DeviceGather, DeviceGemm, DeviceReduction, MultipoleRouting, NearRuleChoice,
};
use nd_fmm_exec::operator::P2pChoice;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::scaling::{HarnessError, Input, Overlap, Settings, Workload, invariance, run};

/// The command line: the workload, the precision and the settings.
struct Arguments {
    workload: Workload,
    n_per_rank: Option<usize>,
    precision: String,
    settings: Settings,
    /// The factors of `--invariance`, if given.
    invariance: Option<Vec<usize>>,
}

/// Parses the options of the module documentation; exits with a message on anything
/// else.
fn arguments() -> Arguments {
    let mut args = Arguments {
        workload: Workload::new(Distribution::Cube, 1_000_000),
        n_per_rank: None,
        precision: "f64".into(),
        settings: Settings::new(8),
        invariance: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--reference" {
            args.settings.reference = true;
            continue;
        }
        if flag == "--kinds" {
            args.settings.kinds = true;
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
        let s = &mut args.settings;
        match flag.as_str() {
            "--dist" => {
                args.workload.distribution = Distribution::from_name(&value)
                    .unwrap_or_else(|| fail(&format!("unknown distribution {value}")));
            }
            "--n" => args.workload.n = number(),
            "--n-per-rank" => args.n_per_rank = Some(number()),
            "--precision" => args.precision = value.clone(),
            "--p" => s.p = number(),
            "--threads" => s.threads = number().max(1),
            "--strategy" => s.strategy = strategy(&flag, &value),
            "--budget" => s.dense_node_budget = Some(number() * 1_000_000),
            "--p2p" => {
                s.p2p = value
                    .parse::<P2pChoice>()
                    .unwrap_or_else(|error| fail(&format!("--p2p: {error}")));
                if let Err(error) = s.p2p.resolve() {
                    fail(&format!("--p2p: {error}"));
                }
            }
            "--mul-add" => {
                s.m2l_mul_add = match value.as_str() {
                    "auto" => None,
                    _ => Some(
                        value
                            .parse()
                            .unwrap_or_else(|error| fail(&format!("--mul-add: {error}"))),
                    ),
                };
            }
            "--reference-strategy" => s.reference_strategy = Some(strategy(&flag, &value)),
            "--overlap" => {
                s.overlap = Overlap::from_name(&value)
                    .unwrap_or_else(|| fail(&format!("--overlap: off, on or both, not {value}")));
            }
            "--input" => {
                s.input = Input::from_name(&value).unwrap_or_else(|| {
                    fail(&format!(
                        "--input: share, rank0, owners or ranges, not {value}"
                    ))
                });
            }
            "--invariance" => {
                let factors = value
                    .split(',')
                    .map(|k| k.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| fail(&format!("--invariance: k[,k…], not {value}")));
                args.invariance = Some(factors);
            }
            "--cut" => {
                s.work_cut = match value.as_str() {
                    "work" => true,
                    "points" => false,
                    _ => fail(&format!("--cut: work or points, not {value}")),
                };
            }
            "--near" => {
                s.near_rule = match value.as_str() {
                    "off" => NearRuleChoice::Off,
                    "cost" => NearRuleChoice::CostModel,
                    pair => match pair
                        .split_once(',')
                        .map(|(w, x)| (w.trim().parse::<f64>(), x.trim().parse::<f64>()))
                    {
                        Some((Ok(w), Ok(x))) => NearRuleChoice::Crossovers { w, x },
                        _ => fail(&format!("--near: off, cost or W,X, not {value}")),
                    },
                };
            }
            "--routing" => {
                s.routing = match value.as_str() {
                    "direct" => MultipoleRouting::Direct,
                    "tree" => MultipoleRouting::default(),
                    tree => match tree.strip_prefix("tree").map(str::parse::<usize>) {
                        Some(Ok(fan_out)) => MultipoleRouting::Tree { fan_out },
                        _ => fail(&format!("--routing: direct, tree or treeF, not {value}")),
                    },
                };
            }
            "--alpha" => {
                let alpha = value
                    .split(',')
                    .map(|a| a.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|alpha| <[f64; 8]>::try_from(alpha).ok())
                    .unwrap_or_else(|| fail(&format!("--alpha: eight numbers, not {value}")));
                s.cost_model = Some(
                    CostModel::new(alpha)
                        .unwrap_or_else(|error| fail(&format!("--alpha: {error}"))),
                );
            }
            "--evals" => s.evaluations = number(),
            "--builds" => s.builds = number().max(1),
            "--leaf-size" => s.max_points_per_leaf = Some(number().max(1)),
            "--errors" => s.vectors = number(),
            "--rhs" => s.rhs = number().max(1),
            "--backend" => {
                s.backend = value.parse().unwrap_or_else(|_| {
                    fail(&format!("--backend: host, cpu, metal or cuda, not {value}"))
                });
            }
            "--sampled" => s.sampled = number(),
            "--device-scratch" => s.device_scratch = Some((number() as u64) << 20),
            "--device-gemm" => {
                s.device_gemm = match value.as_str() {
                    "auto" => DeviceGemm::Auto,
                    "library" => DeviceGemm::Library,
                    "hand" => DeviceGemm::HandWritten,
                    "shared" => DeviceGemm::Shared,
                    _ => fail(&format!(
                        "--device-gemm: auto, library, hand or shared, not {value}"
                    )),
                };
            }
            "--device-gather" => {
                s.device_gather = match value.as_str() {
                    "auto" => DeviceGather::Auto,
                    "separate" => DeviceGather::Separate,
                    "fused" => DeviceGather::Fused,
                    _ => fail(&format!(
                        "--device-gather: auto, separate or fused, not {value}"
                    )),
                };
            }
            "--device-reduction" => {
                s.device_reduction = match value.as_str() {
                    "by-chunk" => DeviceReduction::ByChunk,
                    "walk" => DeviceReduction::Walk,
                    _ => fail(&format!(
                        "--device-reduction: by-chunk or walk, not {value}"
                    )),
                };
            }
            other => fail(&format!("unknown option {other}")),
        }
    }
    if !["f64", "f32"].contains(&args.precision.as_str()) {
        fail(&format!("--precision: f64 or f32, not {}", args.precision));
    }
    args
}

/// The M2L strategy named `value` of option `flag`.
fn strategy(flag: &str, value: &str) -> M2lStrategy {
    match value {
        "auto" => M2lStrategy::Auto,
        "dense" => M2lStrategy::Dense,
        "classes" => M2lStrategy::Classes,
        "rotation" => M2lStrategy::Rotation,
        "batched" => M2lStrategy::Batched,
        "planewave" => M2lStrategy::PlaneWave,
        _ => fail(&format!(
            "{flag}: auto, dense, classes, rotation, batched or planewave, not {value}"
        )),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("scaling: {message}");
    std::process::exit(2);
}

fn main() {
    let mut args = arguments();
    let threading = if args.settings.threads > 1 {
        Threading::Funneled
    } else {
        Threading::Single
    };
    let (universe, provided) =
        mpi::initialize_with_threading(threading).expect("MPI initialises once");
    if provided < threading {
        fail(&format!("MPI provides {provided:?}, {threading:?} needed"));
    }
    let comm = universe.world();
    if let Some(n) = args.n_per_rank {
        args.workload.n = n * comm.size() as usize;
    }
    let status = launch(&args, &comm);
    // MPI is finalised before the exit status is set.
    drop(comm);
    drop(universe);
    std::process::exit(status);
}

/// Runs the launch of `args` on `comm` and prints its report on rank 0. The exit status:
/// 0; 1 if rank-count invariance finds a value that differs; 2 if the harness refuses the
/// arguments (`scaling::HarnessError`, the reason on rank 0). Each is the same on every
/// rank: the refusals depend on the arguments only, and the invariance counts are reduced.
fn launch(args: &Arguments, comm: &SimpleCommunicator) -> i32 {
    let print = |report: &dyn std::fmt::Display| {
        if comm.rank() == 0 {
            println!("{report}");
        }
    };
    let refused = |error: HarnessError| {
        if comm.rank() == 0 {
            eprintln!("scaling: {error}");
        }
        2
    };
    if let Some(factors) = &args.invariance {
        let result = match args.precision.as_str() {
            "f32" => invariance::<f32>(&args.workload, &args.settings, factors, comm),
            _ => invariance::<f64>(&args.workload, &args.settings, factors, comm),
        };
        return match result {
            Ok(report) => {
                print(&report);
                i32::from(report.differing() > 0)
            }
            Err(error) => refused(error),
        };
    }
    let result = match args.precision.as_str() {
        "f32" => run::<f32>(&args.workload, &args.settings, comm),
        _ => run::<f64>(&args.workload, &args.settings, comm),
    };
    match result {
        Ok(report) => {
            print(&report);
            0
        }
        Err(error) => refused(error),
    }
}

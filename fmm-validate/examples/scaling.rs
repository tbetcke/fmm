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
//! On a cluster (docs/design/distributed-fmm.md §11; not run in Phase 5):
//!
//! ```sh
//! mpirun -n 64 --map-by ppr:8:node --bind-to core target/release/examples/scaling \
//!   --dist cube --n 1000000 --precision f64 --p 8 --overlap both
//! ```
//!
//! Options:
//! - `--dist cube|sphere|plummer|clusters` (cube);
//! - `--n N` (10⁶), or `--n-per-rank n` for N = n P (weak scaling);
//! - `--precision f64|f32` (f64), `--p p` (8);
//! - `--threads t` rayon threads per rank (1; MPI at `Threading::Funneled` above 1);
//! - `--strategy auto|dense|classes|rotation`, the M2L strategy (auto);
//! - `--overlap off|on|both` (both: one build, the evaluations alternating, a block each);
//! - `--input share|rank0|owners` (share);
//! - `--evals k` timed evaluations per path (10), `--builds b` (1);
//! - `--errors k` charge vectors of the errors against the direct sum at `--sampled s`
//!   targets (0, not checked; 1,000 targets);
//! - `--reference` the one-rank FMM over every point on rank 0, against the output.

use mpi::Threading;
use mpi::traits::Communicator;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::scaling::{Input, Overlap, Settings, Workload, run};

/// The command line: the workload, the precision and the settings.
struct Arguments {
    workload: Workload,
    n_per_rank: Option<usize>,
    precision: String,
    settings: Settings,
}

/// Parses the options of the module documentation; exits with a message on anything
/// else.
fn arguments() -> Arguments {
    let mut args = Arguments {
        workload: Workload::new(Distribution::Cube, 1_000_000),
        n_per_rank: None,
        precision: "f64".into(),
        settings: Settings::new(8),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--reference" {
            args.settings.reference = true;
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
            "--strategy" => {
                s.strategy = match value.as_str() {
                    "auto" => M2lStrategy::Auto,
                    "dense" => M2lStrategy::Dense,
                    "classes" => M2lStrategy::Classes,
                    "rotation" => M2lStrategy::Rotation,
                    _ => fail(&format!(
                        "--strategy: auto, dense, classes or rotation, not {value}"
                    )),
                };
            }
            "--overlap" => {
                s.overlap = Overlap::from_name(&value)
                    .unwrap_or_else(|| fail(&format!("--overlap: off, on or both, not {value}")));
            }
            "--input" => {
                s.input = Input::from_name(&value).unwrap_or_else(|| {
                    fail(&format!("--input: share, rank0 or owners, not {value}"))
                });
            }
            "--evals" => s.evaluations = number(),
            "--builds" => s.builds = number().max(1),
            "--errors" => s.vectors = number(),
            "--sampled" => s.sampled = number(),
            other => fail(&format!("unknown option {other}")),
        }
    }
    if !["f64", "f32"].contains(&args.precision.as_str()) {
        fail(&format!("--precision: f64 or f32, not {}", args.precision));
    }
    args
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
    let report = match args.precision.as_str() {
        "f32" => run::<f32>(&args.workload, &args.settings, &comm),
        _ => run::<f64>(&args.workload, &args.settings, &comm),
    };
    if comm.rank() == 0 {
        println!("{report}");
    }
}

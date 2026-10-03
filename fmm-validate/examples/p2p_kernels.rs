//! The P2P kernels of `nd-fmm-simd` against `nd_fmm_ref::p2p` (Phase 3S T7, C3S.6).
//! Prints Markdown on stdout and progress on stderr.
//!
//! For each cell of the workloads W1 (FMM-shaped, per-pair and gathered) and W2
//! (all-pairs) of docs/design/simd-p2p.md §8.2, in f32 and f64, with the potential only
//! and with gradients, every kernel runs: the reference `nd_fmm_ref::p2p::p2p`, then
//! `nd_fmm_simd::P2pKernel` on every ISA this machine has. Each is first measured
//! against `direct_sum` on four sets of the cell, then timed. Per cell the report gives
//! the pairs per second, the speed-up over the reference, the fraction of the design
//! §4.6 model as corrected by the T2 spike, the accuracy (largest error relative to the
//! term magnitudes and relative L2), the check of requirement 2, and for per-pair cells
//! whether the per-pair calls give the bits of one gathered call. The workloads, the
//! measures and the model are described in `nd_fmm_validate::p2p_kernels`.
//!
//! Run in release mode with every BLAS thread variable set to 1; the kernels run on one
//! thread. `--quick` runs a reduced set of cells; `--ghz f` sets the clock that turns
//! times into cycles (default: 4.05 on the Apple M3 Max, else none, and the fraction
//! of the model is not printed):
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example p2p_kernels
//! ```
//!
//! Nothing here initialises MPI, although the crate links it (but `--device … --fmm`).
//! Timings are reported, never asserted. The example builds and runs on x86_64 as well,
//! where it times the AVX2 path; Phase 3S took no x86_64 timings.
//!
//! # The device P2P (Phase 4 T6, C4.2; feature `gpu`)
//!
//! `--device b` (with the feature of backend b) times the P2P kernel of `nd-fmm-kernels`
//! instead, on the same workloads made into device calls
//! (`nd_fmm_validate::p2p_device`), against `nd_fmm_simd::P2pKernel` on the same leaves
//! at one thread and at the performance cores (`--threads`, default 12), and checks every
//! cell against `direct_sum` first:
//!
//! - `--device metal`: f32, every candidate layout (the cube layout with 32, 64 and 128
//!   units, the plane layout with 2 and 4 planes per cube); W1 in both forms with
//!   `--leaves` target leaves per launch (default 4,096, a level-sized launch), and W2 at
//!   N = 10³, 10⁴ and 10⁵ (targets in rows of one tile), potential and gradients. Per
//!   cell the pairs per second, the speed-up over the host kernel at 1 and n threads, the
//!   fraction of the C4.2 peak model of device-path.md §13.4 (720 Gpairs/s φ, 480 φ and
//!   ∇φ), and the C4.2 target (25% on W2 at N = 10⁵, 10% on W1 at n_t = 64);
//! - `--device cpu`: f32 and f64, the CPU layout, W1 gathered, at one unit against one
//!   thread (`--leaves` / 4 target leaves, at least 768) and at n units against n
//!   threads (`--leaves`), with the geometric mean of the per-pair time ratio at one
//!   thread against decision 10's target of 1.5;
//! - `--fmm` adds the leaf stage inside the FMM (synchronous stages; the C3.2 cube, and
//!   on a GPU also the Plummer sphere, at p = 3 with gradients, f32, and on the CPU
//!   runtime also f64): the device FMM with P2P on the device, and with P2P on the host
//!   fallback, against the host path at 1 and n threads, the median over `--repeats`
//!   evaluations (default 5); the device runs on a GPU use the n threads for the
//!   host-fallback kinds, as the host row does. It initialises MPI and runs on one rank;
//!   `--only-fmm` runs the leaf stage alone;
//! - `--quick` keeps W1 at n_t = 8 and 64 and W2 at N = 10³ and 10⁵.
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --features metal --example p2p_kernels -- \
//!     --device metal --fmm
//! cargo run --release -p nd-fmm-validate --features cpu --example p2p_kernels -- \
//!     --device cpu --fmm
//! ```
//!
//! Metal needs a process with GPU access (outside the macOS sandbox). Device timings
//! queue many launches between syncs and exclude compilation (a warm-up launch).

use nd_fmm_simd::{Isa, SimdScalar};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::p2p_kernels::{
    Accuracy, CHECKED_SETS, Cell, Form, Kernel, M3_MAX_GHZ, Model, Outputs, POOL, check, model,
    oracles, passes, per_pair_equals_gathered, precision_name, time_kernel,
};

/// The BLAS and threading variables the header reports.
const THREAD_VARIABLES: [&str; 6] = [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
];

/// One measured kernel on one cell.
struct Row {
    cell: Cell,
    precision: &'static str,
    gradients: bool,
    kernel: String,
    isa: Option<Isa>,
    pairs_per_second: f64,
    accuracy: Accuracy,
    passed: bool,
    /// Per-pair cells of `nd-fmm-simd`: whether the per-pair calls give the bits of one
    /// gathered call.
    per_pair_bits: Option<bool>,
    model: Option<Model>,
}

/// The command line.
struct Arguments {
    quick: bool,
    ghz: Option<f64>,
    /// `--device b`: the device rows instead of the host table.
    device: Option<String>,
    /// `--fmm`: with `--device`, also the leaf stage inside the FMM.
    fmm: bool,
    /// `--only-fmm`: with `--device`, the leaf stage alone.
    only_fmm: bool,
    /// `--leaves n`: target leaves per device launch of W1.
    leaves: usize,
    /// `--threads n`: the host threads and CPU-runtime units of the all-cores rows.
    threads: usize,
    /// `--repeats r`: evaluations per FMM configuration.
    repeats: usize,
}

/// Parses `--quick`, `--ghz f`, `--device b`, `--fmm`, `--leaves n`, `--threads n` and
/// `--repeats r`; exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: p2p_kernels [--quick] [--ghz f] [--device cpu|metal|cuda [--fmm|--only-fmm] \
             [--leaves n] [--threads n] [--repeats r]], f > 0, n, r >= 1; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        quick: false,
        ghz: cpu_model().contains("Apple M3 Max").then_some(M3_MAX_GHZ),
        device: None,
        fmm: false,
        only_fmm: false,
        leaves: 4096,
        threads: 12,
        repeats: 5,
    };
    let positive = |value: &str| -> usize {
        match value.parse::<usize>() {
            Ok(n) if n >= 1 => n,
            _ => usage(),
        }
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--quick", _) => {
                parsed.quick = true;
                tail
            }
            ("--ghz", [value, tail @ ..]) => {
                match value.parse::<f64>() {
                    Ok(f) if f > 0.0 => parsed.ghz = Some(f),
                    _ => usage(),
                }
                tail
            }
            ("--device", [value, tail @ ..]) => {
                parsed.device = Some(value.clone());
                tail
            }
            ("--fmm", _) => {
                parsed.fmm = true;
                tail
            }
            ("--only-fmm", _) => {
                parsed.fmm = true;
                parsed.only_fmm = true;
                tail
            }
            ("--leaves", [value, tail @ ..]) => {
                parsed.leaves = positive(value);
                tail
            }
            ("--threads", [value, tail @ ..]) => {
                parsed.threads = positive(value);
                tail
            }
            ("--repeats", [value, tail @ ..]) => {
                parsed.repeats = positive(value);
                tail
            }
            _ => usage(),
        };
    }
    if parsed.fmm && parsed.device.is_none() {
        usage();
    }
    parsed
}

fn main() {
    let arguments = arguments();
    if let Some(name) = &arguments.device {
        #[cfg(feature = "gpu")]
        {
            device::main(name, &arguments);
            return;
        }
        #[cfg(not(feature = "gpu"))]
        {
            eprintln!(
                "--device {name}: build with the backend's feature, e.g. --features cpu or \
                 --features metal"
            );
            std::process::exit(2);
        }
    }
    let Arguments { quick, ghz, .. } = arguments;
    let cells = if quick { Cell::quick() } else { Cell::all() };
    let mut rows = run::<f32>(&cells);
    rows.extend(run::<f64>(&cells));
    print_header(quick, ghz);
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            print_table(&rows, precision, gradients, ghz);
        }
    }
    print_summary(&rows, ghz);
}

/// Checks and times every kernel of precision `T` on every cell, with and without
/// gradients.
fn run<T: SimdScalar>(cells: &[Cell]) -> Vec<Row> {
    let mut rows = Vec::new();
    for &cell in cells {
        let pool = cell.pool::<T>();
        let oracles = oracles(&pool);
        for gradients in [false, true] {
            eprintln!(
                "{} {} {}",
                precision_name::<T>(),
                cell.label(),
                if gradients { "φ, ∇φ" } else { "φ" }
            );
            let mut reference = Accuracy::default();
            for kernel in Kernel::<T>::all() {
                let accuracy = check(&kernel, &pool, &oracles, cell.form(), gradients);
                if kernel.isa().is_none() {
                    reference = accuracy;
                }
                let per_pair_bits = (kernel.isa().is_some() && cell.form() == Form::PerPair)
                    .then(|| per_pair_equals_gathered(&kernel, &pool[0], gradients));
                let mut outputs = Outputs::for_pool(&pool);
                let pairs_per_second =
                    time_kernel(&kernel, &pool, &mut outputs, cell.form(), gradients);
                rows.push(Row {
                    cell,
                    precision: precision_name::<T>(),
                    gradients,
                    kernel: kernel.name(),
                    isa: kernel.isa(),
                    pairs_per_second,
                    accuracy,
                    passed: passes::<T>(accuracy, reference),
                    per_pair_bits,
                    model: kernel.isa().and_then(|isa| model::<T>(isa, gradients)),
                });
            }
        }
    }
    rows
}

/// The machine, the toolchain, the ISAs and the method.
fn print_header(quick: bool, ghz: Option<f64>) {
    println!("# P2P kernels: nd-fmm-simd against nd_fmm_ref::p2p (Phase 3S T7)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| CPU | {} |", cpu_model());
    println!(
        "| cores | {}{} |",
        cores(),
        performance_cores().map_or(String::new(), |p| format!(", {p} performance"))
    );
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!(
        "| ISAs available | {} (detected: {}) |",
        Isa::available()
            .map(|isa| isa.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        Isa::detect()
    );
    println!(
        "| thread variables | {} |",
        THREAD_VARIABLES
            .iter()
            .map(|k| format!(
                "{k}={}",
                std::env::var(k).unwrap_or_else(|_| "unset".into())
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "| clock for cycles | {} |",
        ghz.map_or("none (pass --ghz)".to_string(), |f| format!(
            "{f} GHz assumed"
        ))
    );
    println!(
        "| mode | {} |",
        if quick {
            "--quick (reduced set of cells)"
        } else {
            "full"
        }
    );
    println!("| timing | median of 15 batches of ≥ 20 ms, one thread |");
    println!();
    println!(
        "- Workloads (design §8.2): W1, a target leaf of n_t points in [-1, 1]^3 and the \
         27 leaves of its 3 x 3 x 3 block, n_t sources each (the centre leaf equal to the \
         targets), a pool of {POOL} sets cycled; per-pair = 27 calls, gathered = one call. \
         W2, N sources in [0, 1]^3, targets distinct (t≠s) or equal (t=s). Charges in \
         [-1, 1]. The inputs of the T2 spike."
    );
    println!(
        "- Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. \
         \"x ref\": pairs per second over the reference's on the same cell."
    );
    println!(
        "- Accuracy against `direct_sum` on the first {CHECKED_SETS} sets of the cell: \
         \"max\" is the largest error relative to the term magnitudes of its target \
         (Σ|q|/r for φ, Σ|q|/r² per component of ∇φ), \"L2\" the relative L2 error. \
         Check: requirement 2 (design §3) as decided in T5, max within 1e-6 (f32) or \
         1e-14 (f64), or within twice the reference's error on the same inputs."
    );
    println!(
        "- Model: pairs per cycle of design §4.6 as corrected by the T2 spike (NEON: \
         4 FP pipes and the divider, W / max(ops/4, 3); AVX2: 2 FMA pipes, 2W / ops), \
         with the measured pairs per cycle at the assumed clock."
    );
    println!(
        "- The reference's speed is bimodal on the M3 Max (T2 spike, \"Other findings\"): \
         potential-only cells run at about 0.34 or about 0.86 Gpairs/s on the same data, \
         depending on the run, because its compiled loop keeps φ in memory across the \
         source loop. The speed-ups use the measured values as they are."
    );
    println!();
}

/// Gpairs/s with three decimals.
fn gpairs(r: &Row) -> String {
    format!("{:.3}", r.pairs_per_second / 1e9)
}

/// The rows of one precision and output, cell by cell.
fn print_table(rows: &[Row], precision: &str, gradients: bool, ghz: Option<f64>) {
    println!(
        "## {precision}, {}",
        if gradients {
            "potential and gradient"
        } else {
            "potential"
        }
    );
    println!();
    println!(
        "| cell | kernel | Gpairs/s | x ref | model ops | pairs/cycle | of model | max φ | L2 φ | max ∇φ | L2 ∇φ | check | per-pair = gathered |"
    );
    println!(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
    );
    let mine: Vec<&Row> = rows
        .iter()
        .filter(|r| r.precision == precision && r.gradients == gradients)
        .collect();
    for r in &mine {
        let reference = mine
            .iter()
            .find(|q| q.cell == r.cell && q.isa.is_none())
            .expect("the reference runs on every cell");
        let cycles = ghz.map(|f| r.pairs_per_second / (f * 1e9));
        println!(
            "| {} | {} | {} | {:.2} | {} | {} | {} | {:.1e} | {:.1e} | {} | {} | {} | {} |",
            r.cell.label(),
            r.kernel,
            gpairs(r),
            r.pairs_per_second / reference.pairs_per_second,
            r.model.map_or("–".into(), |m| m.ops.to_string()),
            cycles.map_or("–".into(), |c| format!("{c:.3}")),
            match (r.model, cycles) {
                (Some(m), Some(c)) => format!("{:.0}%", 100.0 * c / m.pairs_per_cycle),
                _ => "–".into(),
            },
            r.accuracy.max_potential,
            r.accuracy.l2_potential,
            if gradients {
                format!("{:.1e}", r.accuracy.max_gradient)
            } else {
                String::new()
            },
            if gradients {
                format!("{:.1e}", r.accuracy.l2_gradient)
            } else {
                String::new()
            },
            if r.passed { "pass" } else { "FAIL" },
            r.per_pair_bits.map_or("", |b| if b { "yes" } else { "NO" }),
        );
    }
    println!();
}

/// The geometric mean of positive numbers.
fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// Geometric means over the W1 cells of each form and over W2, per kernel.
fn print_summary(rows: &[Row], ghz: Option<f64>) {
    println!("## Summary");
    println!();
    println!(
        "Geometric means over the cells of each group: Gpairs/s, the speed-up over the \
         reference, and for W1 gathered the fraction of the model. Worst accuracy over \
         the cells of all groups, and the checks passed."
    );
    println!();
    println!(
        "| precision | output | kernel | W1 per-pair Gpairs/s (x ref) | W1 gathered Gpairs/s (x ref) | W2 Gpairs/s (x ref) | W1 gathered of model | worst max φ | worst max ∇φ | checks |"
    );
    println!("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
    type Group = fn(Cell) -> bool;
    let groups: [Group; 3] = [
        |c| {
            matches!(
                c,
                Cell::W1 {
                    form: Form::PerPair,
                    ..
                }
            )
        },
        |c| {
            matches!(
                c,
                Cell::W1 {
                    form: Form::Gathered,
                    ..
                }
            )
        },
        |c| matches!(c, Cell::W2 { .. }),
    ];
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            let mine: Vec<&Row> = rows
                .iter()
                .filter(|r| r.precision == precision && r.gradients == gradients)
                .collect();
            let mut kernels: Vec<&str> = Vec::new();
            for r in &mine {
                if !kernels.contains(&r.kernel.as_str()) {
                    kernels.push(&r.kernel);
                }
            }
            for kernel in kernels {
                let of = |r: &&&Row| r.kernel == kernel;
                let ratio = |r: &Row| {
                    let reference = mine
                        .iter()
                        .find(|q| q.cell == r.cell && q.isa.is_none())
                        .expect("the reference runs on every cell");
                    r.pairs_per_second / reference.pairs_per_second
                };
                let group = |g: Group| -> String {
                    let v: Vec<&&Row> = mine.iter().filter(of).filter(|r| g(r.cell)).collect();
                    if v.is_empty() {
                        return "–".into();
                    }
                    let speed: Vec<f64> = v.iter().map(|r| r.pairs_per_second).collect();
                    let x: Vec<f64> = v.iter().map(|r| ratio(r)).collect();
                    format!("{:.3} ({:.2})", geomean(&speed) / 1e9, geomean(&x))
                };
                let gathered: Vec<f64> = mine
                    .iter()
                    .filter(of)
                    .filter(|r| groups[1](r.cell))
                    .filter_map(|r| {
                        Some(r.pairs_per_second / (ghz? * 1e9 * r.model?.pairs_per_cycle))
                    })
                    .collect();
                let all: Vec<&&Row> = mine.iter().filter(of).collect();
                let worst = all
                    .iter()
                    .fold(Accuracy::default(), |a, r| a.worst(r.accuracy));
                println!(
                    "| {precision} | {} | {kernel} | {} | {} | {} | {} | {:.1e} | {} | {}/{} |",
                    if gradients { "φ, ∇φ" } else { "φ" },
                    group(groups[0]),
                    group(groups[1]),
                    group(groups[2]),
                    if gathered.is_empty() {
                        "–".into()
                    } else {
                        format!("{:.0}%", 100.0 * geomean(&gathered))
                    },
                    worst.max_potential,
                    if gradients {
                        format!("{:.1e}", worst.max_gradient)
                    } else {
                        String::new()
                    },
                    all.iter().filter(|r| r.passed).count(),
                    all.len()
                );
            }
        }
    }
    println!();
}

/// The device rows (module documentation, "The device P2P").
#[cfg(feature = "gpu")]
mod device {
    use std::time::Duration;

    use mpi::Threading;
    use mpi::topology::SimpleCommunicator;
    use mpi::traits::Equivalence;
    use nd_fmm_exec::fmm::{Backend, DeviceP2pLayout, FmmBuilder, OperatorKind, Output};
    use nd_fmm_kernels::p2p::P2pLayout;
    use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
    use nd_fmm_math::RealScalar;
    use nd_fmm_simd::{P2pKernel, SimdScalar};
    use nd_fmm_tables::cache::Stored;
    use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};
    use nd_fmm_validate::p2p_device::{
        DeviceProblem, seconds_per_all_pairs, seconds_per_evaluation,
    };
    use nd_fmm_validate::p2p_kernels::{
        Accuracy, CHECKED_SETS, Form, Kernel, Oracle, Set, W1_SEED, W1_TARGETS, W2_SEED, check,
        oracles, passes, precision_name, w1, w2,
    };

    use super::{Arguments, cores, cpu_model, geomean, performance_cores, target, toolchain};

    /// The C4.2 peak model of device-path.md §13.4, pairs per second: φ, then φ and ∇φ.
    const MODEL: [f64; 2] = [720e9, 480e9];

    /// The C4.2 targets, fractions of the model: W2 at N = 10⁵ and W1 at n_t = 64.
    const TARGET_W2: f64 = 0.25;
    const TARGET_W1: f64 = 0.10;

    /// The W2 sizes on a GPU.
    const GPU_W2: [usize; 3] = [1000, 10_000, 100_000];

    /// The targets of a W2 set the accuracy is measured at.
    const W2_CHECKED: usize = 1000;

    /// One timed device cell.
    struct Row {
        cell: String,
        precision: &'static str,
        gradients: bool,
        layout: P2pLayout,
        rate: f64,
        host_one: f64,
        host_many: f64,
        accuracy: Accuracy,
        passed: bool,
        /// For the CPU layout: the all-cores rate and its host counterpart.
        all_cores: Option<(f64, f64)>,
    }

    pub fn main(name: &str, arguments: &Arguments) {
        let Some(kind) = BackendKind::from_name(name) else {
            eprintln!("--device {name}: expected cpu, metal or cuda");
            std::process::exit(2);
        };
        let mut device = Device::open(kind).unwrap_or_else(|error| {
            eprintln!("--device {name}: {error} (Metal needs a process with GPU access)");
            std::process::exit(1);
        });
        print_header(&device, arguments);
        let mut rows = Vec::new();
        match kind {
            _ if arguments.only_fmm => {}
            BackendKind::Cpu => {
                rows.extend(cpu_rows::<f32>(&mut device, arguments));
                rows.extend(cpu_rows::<f64>(&mut device, arguments));
                print_cpu(&rows, arguments.threads);
            }
            _ => {
                rows.extend(gpu_rows::<f32>(&mut device, arguments));
                print_gpu(&rows, arguments.threads);
            }
        }
        if arguments.fmm {
            leaf_stage(kind, arguments);
        }
    }

    fn print_header(device: &Device, arguments: &Arguments) {
        let info = device.info();
        println!("# The device P2P against nd-fmm-simd (Phase 4 T6, C4.2)");
        println!();
        println!("| item | value |");
        println!("| --- | --- |");
        println!("| machine | {} ({}) |", cpu_model(), cores());
        println!(
            "| performance cores | {} |",
            performance_cores().map_or("unknown".into(), |p| p.to_string())
        );
        println!("| target | {} |", target());
        println!("| toolchain | {} |", toolchain());
        println!("| device | {info} |");
        println!(
            "| host kernel | nd-fmm-simd on {} (detected) |",
            nd_fmm_simd::Isa::detect()
        );
        println!(
            "| W1 leaves per launch | {} (one-thread rows: {}) |",
            arguments.leaves,
            one_thread_leaves(arguments)
        );
        println!("| threads of the all-cores rows | {} |", arguments.threads);
        println!(
            "| timing | median of 15 batches of >= 20 ms; device launches queued between \
             syncs, a warm-up launch first (compilation excluded) |"
        );
        println!();
    }

    /// The target leaves of the one-thread rows: a quarter of `--leaves`, at least 768
    /// (T3's launches).
    fn one_thread_leaves(arguments: &Arguments) -> usize {
        (arguments.leaves / 4).max(768)
    }

    /// The W1 leaf sizes: every one, or with `--quick` 8 and 64.
    fn w1_sizes(arguments: &Arguments) -> Vec<usize> {
        if arguments.quick {
            vec![8, 64]
        } else {
            W1_TARGETS.to_vec()
        }
    }

    /// The layouts a GPU runs: the cube layout with 32, 64 (the default) and 128 units,
    /// the plane layout with 2 and 4 planes per cube.
    fn gpu_layouts(device: &Device, precision: Precision) -> Vec<P2pLayout> {
        [
            P2pLayout::Cube { units: 64 },
            P2pLayout::Cube { units: 32 },
            P2pLayout::Cube { units: 128 },
            P2pLayout::Plane { planes: 2 },
            P2pLayout::Plane { planes: 4 },
        ]
        .into_iter()
        .filter(|l| l.check(device.info(), precision).is_ok())
        .collect()
    }

    /// Every GPU cell: W1 in both forms, W2 at each N, φ and φ with ∇φ, each layout.
    fn gpu_rows<T: SimdScalar + DeviceFloat + RealScalar>(
        device: &mut Device,
        arguments: &Arguments,
    ) -> Vec<Row> {
        let mut rows = Vec::new();
        let layouts = gpu_layouts(device, T::FLOAT);
        let simd = Kernel::Simd(P2pKernel::<T>::detect());
        for n_t in w1_sizes(arguments) {
            let pool = w1::<T>(n_t, W1_SEED);
            let checked = oracles(&pool);
            let with_rows: Vec<(Oracle, std::ops::Range<usize>)> = checked
                .iter()
                .enumerate()
                .map(|(i, o)| (o.clone(), i..i + 1))
                .collect();
            let pairs_per_set = pool[0].pairs() as f64;
            for form in [Form::PerPair, Form::Gathered] {
                for gradients in [false, true] {
                    let cell = format!("W1 n_t={n_t} {}", form.name());
                    eprintln!("{} {cell} gradients {gradients}", precision_name::<T>());
                    let reference = check(&Kernel::Reference, &pool, &checked, form, gradients);
                    let one = one_thread_leaves(arguments);
                    let host_one = pairs_per_set * one as f64
                        / seconds_per_evaluation(&simd, &pool, form, gradients, one, 1);
                    let host_many = pairs_per_set * arguments.leaves as f64
                        / seconds_per_evaluation(
                            &simd,
                            &pool,
                            form,
                            gradients,
                            arguments.leaves,
                            arguments.threads,
                        );
                    let mut problem =
                        DeviceProblem::w1(device, &pool, form, arguments.leaves, gradients)
                            .expect("the W1 problem uploads");
                    for &layout in &layouts {
                        let accuracy = problem
                            .accuracy(device, layout, &with_rows)
                            .expect("the device evaluates");
                        let seconds = problem
                            .seconds_per_launch(device, layout)
                            .expect("the device evaluates");
                        rows.push(Row {
                            cell: cell.clone(),
                            precision: precision_name::<T>(),
                            gradients,
                            layout,
                            rate: problem.pairs() as f64 / seconds,
                            host_one,
                            host_many,
                            accuracy,
                            passed: passes::<T>(accuracy, reference),
                            all_cores: None,
                        });
                    }
                }
            }
        }
        let w2_sizes: &[usize] = if arguments.quick {
            &[1000, 100_000]
        } else {
            &GPU_W2
        };
        for &n in w2_sizes {
            for equal in [false, true] {
                let set = w2::<T>(n, equal, W2_SEED);
                let m = n.min(W2_CHECKED);
                let subset = Set {
                    targets: set.targets[..m].to_vec(),
                    leaves: Vec::new(),
                    sources: set.sources.clone(),
                    charges: set.charges.clone(),
                };
                let oracle = Oracle::new(&subset);
                let pairs = set.pairs() as f64;
                for gradients in [false, true] {
                    let cell = format!("W2 N={n} {}", if equal { "t=s" } else { "t≠s" });
                    eprintln!("{} {cell} gradients {gradients}", precision_name::<T>());
                    let reference = check(
                        &Kernel::Reference,
                        std::slice::from_ref(&subset),
                        std::slice::from_ref(&oracle),
                        Form::Gathered,
                        gradients,
                    );
                    let host_one = pairs / seconds_per_all_pairs(&simd, &set, gradients, 1);
                    let host_many =
                        pairs / seconds_per_all_pairs(&simd, &set, gradients, arguments.threads);
                    for &layout in &layouts {
                        let chunk = layout.tile(device.info(), T::FLOAT);
                        let mut problem = DeviceProblem::w2(device, &set, chunk, gradients)
                            .expect("the W2 problem uploads");
                        let accuracy = problem
                            .accuracy(device, layout, &[(oracle.clone(), 0..m.div_ceil(chunk))])
                            .expect("the device evaluates");
                        let seconds = problem
                            .seconds_per_launch(device, layout)
                            .expect("the device evaluates");
                        rows.push(Row {
                            cell: cell.clone(),
                            precision: precision_name::<T>(),
                            gradients,
                            layout,
                            rate: problem.pairs() as f64 / seconds,
                            host_one,
                            host_many,
                            accuracy,
                            passed: passes::<T>(accuracy, reference),
                            all_cores: None,
                        });
                    }
                }
            }
        }
        rows
    }

    /// The CPU layout on W1 gathered: one unit against one thread, and n units against
    /// n threads.
    fn cpu_rows<T: SimdScalar + DeviceFloat + RealScalar>(
        device: &mut Device,
        arguments: &Arguments,
    ) -> Vec<Row> {
        let mut rows = Vec::new();
        let simd = Kernel::Simd(P2pKernel::<T>::detect());
        let form = Form::Gathered;
        let layout = P2pLayout::default_for(device.info());
        let one = one_thread_leaves(arguments);
        for n_t in w1_sizes(arguments) {
            let pool = w1::<T>(n_t, W1_SEED);
            let checked = oracles(&pool);
            let with_rows: Vec<(Oracle, std::ops::Range<usize>)> = checked
                .iter()
                .enumerate()
                .map(|(i, o)| (o.clone(), i..i + 1))
                .collect();
            let pairs_per_set = pool[0].pairs() as f64;
            for gradients in [false, true] {
                eprintln!(
                    "{} W1 n_t={n_t} gathered gradients {gradients}",
                    precision_name::<T>()
                );
                let reference = check(&Kernel::Reference, &pool, &checked, form, gradients);
                // One unit against one thread.
                device.limit_units(1);
                let mut problem = DeviceProblem::w1(device, &pool, form, one, gradients)
                    .expect("the W1 problem uploads");
                let accuracy = problem
                    .accuracy(device, layout, &with_rows)
                    .expect("the device evaluates");
                let rate = problem.pairs() as f64
                    / problem
                        .seconds_per_launch(device, layout)
                        .expect("the device evaluates");
                drop(problem);
                let host_one = pairs_per_set * one as f64
                    / seconds_per_evaluation(&simd, &pool, form, gradients, one, 1);
                // n units against n threads, a level-sized launch.
                device.limit_units(arguments.threads as u32);
                let mut problem =
                    DeviceProblem::w1(device, &pool, form, arguments.leaves, gradients)
                        .expect("the W1 problem uploads");
                let all = problem.pairs() as f64
                    / problem
                        .seconds_per_launch(device, layout)
                        .expect("the device evaluates");
                drop(problem);
                let host_many = pairs_per_set * arguments.leaves as f64
                    / seconds_per_evaluation(
                        &simd,
                        &pool,
                        form,
                        gradients,
                        arguments.leaves,
                        arguments.threads,
                    );
                rows.push(Row {
                    cell: format!("W1 n_t={n_t} gathered"),
                    precision: precision_name::<T>(),
                    gradients,
                    layout,
                    rate,
                    host_one,
                    host_many,
                    accuracy,
                    passed: passes::<T>(accuracy, reference),
                    all_cores: Some((all, host_many)),
                });
            }
        }
        device.limit_units(nd_fmm_kernels::CPU_MAX_UNITS);
        rows
    }

    fn gpairs(rate: f64) -> String {
        format!("{:.3}", rate / 1e9)
    }

    fn accuracy_columns(r: &Row) -> String {
        format!(
            "{:.1e} | {} | {}",
            r.accuracy.max_potential,
            if r.gradients {
                format!("{:.1e}", r.accuracy.max_gradient)
            } else {
                "–".into()
            },
            if r.passed { "pass" } else { "FAIL" }
        )
    }

    fn print_cpu(rows: &[Row], threads: usize) {
        println!("## The CPU layout against nd-fmm-simd, W1 gathered");
        println!();
        println!(
            "Time ratio: the per-pair time of the device over that of nd-fmm-simd on the \
             same leaves (rates inverted); decision 10's target is a geometric mean of at most \
             1.5 at one thread. Accuracy against `direct_sum` on {CHECKED_SETS} sets: the \
             largest error relative to the term magnitudes, and the check of the contract \
             (1e-6 / 1e-14, or twice the reference's error)."
        );
        println!();
        println!(
            "| precision | output | cell | device 1 unit Gpairs/s | simd 1 thread | ratio | \
             device {threads} units | simd {threads} threads | ratio | max φ | max ∇φ | check |"
        );
        println!(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        );
        let mut one = Vec::new();
        let mut many = Vec::new();
        for r in rows {
            let (all, host_all) = r.all_cores.expect("CPU rows");
            one.push(r.host_one / r.rate);
            many.push(host_all / all);
            println!(
                "| {} | {} | {} | {} | {} | {:.3} | {} | {} | {:.3} | {} |",
                r.precision,
                if r.gradients { "φ, ∇φ" } else { "φ" },
                r.cell,
                gpairs(r.rate),
                gpairs(r.host_one),
                r.host_one / r.rate,
                gpairs(all),
                gpairs(host_all),
                host_all / all,
                accuracy_columns(r)
            );
        }
        println!();
        println!(
            "Geometric mean of the time ratio over the {} cells: **one thread {:.3}** \
             (target ≤ 1.5: {}), all {threads} cores {:.3} (reported, not targeted). Ranges: \
             one thread {:.2}–{:.2}, all cores {:.2}–{:.2}. Checks passed: {}/{}.",
            rows.len(),
            geomean(&one),
            if geomean(&one) <= 1.5 {
                "met"
            } else {
                "MISSED"
            },
            geomean(&many),
            one.iter().copied().fold(f64::INFINITY, f64::min),
            one.iter().copied().fold(0.0, f64::max),
            many.iter().copied().fold(f64::INFINITY, f64::min),
            many.iter().copied().fold(0.0, f64::max),
            rows.iter().filter(|r| r.passed).count(),
            rows.len()
        );
        println!();
    }

    fn print_gpu(rows: &[Row], threads: usize) {
        println!("## The GPU layouts, f32");
        println!();
        println!(
            "Model: the C4.2 peak model of device-path.md §13.4, 720 Gpairs/s (φ) and 480 \
             (φ, ∇φ), from 40 cores × 128 lanes × 1.4 GHz and 10 or 15 operations per pair. \
             Speed-ups over nd-fmm-simd on the same cell at 1 and {threads} threads. W1 rows \
             launch the stated leaves per launch; W2 rows split the targets into rows of one \
             tile."
        );
        println!();
        for gradients in [false, true] {
            println!(
                "### {}",
                if gradients {
                    "potential and gradient"
                } else {
                    "potential"
                }
            );
            println!();
            println!(
                "| cell | layout | Gpairs/s | of model | x simd 1 thread | x simd {threads} \
                 threads | max φ | max ∇φ | check |"
            );
            println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
            for r in rows.iter().filter(|r| r.gradients == gradients) {
                println!(
                    "| {} | {} | {} | {:.1}% | {:.1} | {:.2} | {} |",
                    r.cell,
                    r.layout,
                    gpairs(r.rate),
                    100.0 * r.rate / MODEL[usize::from(gradients)],
                    r.rate / r.host_one,
                    r.rate / r.host_many,
                    accuracy_columns(r)
                );
            }
            println!();
        }
        println!("## The C4.2 target (measured, never asserted)");
        println!();
        println!("| cell | output | target | best layout | Gpairs/s | of model | met |");
        println!("| --- | --- | ---: | --- | ---: | ---: | --- |");
        for (prefix, target) in [("W2 N=100000", TARGET_W2), ("W1 n_t=64 ", TARGET_W1)] {
            for gradients in [false, true] {
                let model = MODEL[usize::from(gradients)];
                for r in rows
                    .iter()
                    .filter(|r| r.gradients == gradients && r.cell.starts_with(prefix))
                {
                    let best = rows
                        .iter()
                        .filter(|q| q.gradients == gradients && q.cell == r.cell)
                        .max_by(|a, b| a.rate.total_cmp(&b.rate))
                        .unwrap();
                    if !std::ptr::eq(best, r) {
                        continue;
                    }
                    let default = rows
                        .iter()
                        .find(|q| {
                            q.gradients == gradients
                                && q.cell == r.cell
                                && q.layout == P2pLayout::Cube { units: 64 }
                        })
                        .unwrap();
                    println!(
                        "| {} | {} | {:.0}% ({:.0} Gpairs/s) | {} (default cube 64: {:.1}%) | \
                         {} | {:.1}% | {} |",
                        r.cell,
                        if gradients { "φ, ∇φ" } else { "φ" },
                        100.0 * target,
                        target * model / 1e9,
                        r.layout,
                        100.0 * default.rate / model,
                        gpairs(r.rate),
                        100.0 * r.rate / model,
                        if r.rate >= target * model {
                            "yes"
                        } else {
                            "NO"
                        }
                    );
                }
            }
        }
        println!();
    }

    /// One FMM configuration of the leaf-stage table.
    struct Stage {
        problem: &'static str,
        precision: &'static str,
        configuration: String,
        leaf_stage: f64,
        evaluation: f64,
        difference: Option<(f64, f64)>,
    }

    /// The relative L2 difference of `got` from `want`, φ and ∇φ.
    fn difference<T: RealScalar>(got: &Output<T>, want: &Output<T>) -> (f64, f64) {
        let l2 = |a: &[T], b: &[T]| {
            let (mut d, mut r) = (0.0f64, 0.0f64);
            for (&x, &y) in a.iter().zip(b) {
                let (x, y) = (x.to_f64(), y.to_f64());
                d += (x - y).powi(2);
                r += y * y;
            }
            (d / r).sqrt()
        };
        (
            l2(&got.potential, &want.potential),
            l2(
                got.gradient.as_ref().unwrap().as_flattened(),
                want.gradient.as_ref().unwrap().as_flattened(),
            ),
        )
    }

    /// Builds `builder`, evaluates once (warm-up), then `repeats` times; the median leaf
    /// stage and evaluation in seconds, and the last output.
    fn measure<T: Stored + SimdScalar + Equivalence + Default>(
        builder: &FmmBuilder<T>,
        points: &[[f64; 3]],
        charges: &[T],
        repeats: usize,
        comm: &SimpleCommunicator,
    ) -> (f64, f64, Output<T>) {
        let mut fmm = builder
            .build(points, points, comm)
            .unwrap_or_else(|error| panic!("the FMM does not build: {error}"));
        let mut output = fmm.evaluate(charges).expect("the FMM evaluates");
        let mut leaves = Vec::new();
        let mut totals = Vec::new();
        for _ in 0..repeats {
            output = fmm.evaluate(charges).expect("the FMM evaluates");
            leaves.push(output.timings.evaluate_leaves);
            totals.push(output.timings.total());
        }
        let median = |mut v: Vec<Duration>| {
            v.sort();
            v[v.len() / 2].as_secs_f64()
        };
        (median(leaves), median(totals), output)
    }

    /// The leaf stage of the FMM in `T` on `problem` (`config`), every configuration.
    fn stages<T: Stored + SimdScalar + Equivalence + Default + RealScalar>(
        kind: BackendKind,
        (name, config): (&'static str, Config),
        arguments: &Arguments,
        comm: &SimpleCommunicator,
        out: &mut Vec<Stage>,
    ) {
        let problem = Problem::new(&config);
        let charges: Vec<T> = problem.charges[0].iter().map(|&q| T::from_f64(q)).collect();
        let builder = FmmBuilder::<T>::new(3)
            .gradients(true)
            .max_level(config.max_level)
            .max_points_per_leaf(config.max_points_per_leaf);
        let backend = match kind {
            BackendKind::Cpu => Backend::Cpu,
            BackendKind::Metal => Backend::Metal,
            BackendKind::Cuda => Backend::Cuda,
        };
        let precision = precision_name::<T>();
        let n = arguments.threads;
        let (leaf, total, host) = measure(
            &builder.clone().threads(1),
            &problem.points,
            &charges,
            arguments.repeats,
            comm,
        );
        out.push(Stage {
            problem: name,
            precision,
            configuration: "host, 1 thread".into(),
            leaf_stage: leaf,
            evaluation: total,
            difference: None,
        });
        let (leaf, total, _) = measure(
            &builder.clone().threads(n),
            &problem.points,
            &charges,
            arguments.repeats,
            comm,
        );
        out.push(Stage {
            problem: name,
            precision,
            configuration: format!("host, {n} threads"),
            leaf_stage: leaf,
            evaluation: total,
            difference: None,
        });
        let device = builder.clone().backend(backend).synchronous_stages(true);
        let mut configurations: Vec<(String, FmmBuilder<T>)> = Vec::new();
        match kind {
            BackendKind::Cpu => {
                for units in [1, n] {
                    configurations.push((
                        format!("{backend}, P2P on the device (cpu layout), {units} unit(s)"),
                        device.clone().threads(units),
                    ));
                }
                configurations.push((
                    format!("{backend}, P2P on the host fallback, 1 thread"),
                    device.clone().host_fallback([OperatorKind::P2p]),
                ));
            }
            _ => {
                // The pool of n threads serves the host-fallback kinds (L2P, M2P).
                for (label, layout) in [
                    ("cube 64", DeviceP2pLayout::Auto),
                    ("plane 2", DeviceP2pLayout::Plane(2)),
                ] {
                    configurations.push((
                        format!("{backend}, P2P on the device ({label}), {n} threads"),
                        device.clone().threads(n).device_p2p_layout(layout),
                    ));
                }
                configurations.push((
                    format!("{backend}, P2P on the host fallback, {n} threads"),
                    device.clone().threads(n).host_fallback([OperatorKind::P2p]),
                ));
            }
        }
        for (configuration, builder) in configurations {
            eprintln!("FMM leaf stage: {name}, {precision}, {configuration}");
            let (leaf, total, output) =
                measure(&builder, &problem.points, &charges, arguments.repeats, comm);
            out.push(Stage {
                problem: name,
                precision,
                configuration,
                leaf_stage: leaf,
                evaluation: total,
                difference: Some(difference(&output, &host)),
            });
        }
    }

    /// The leaf stage inside the FMM (module documentation).
    fn leaf_stage(kind: BackendKind, arguments: &Arguments) {
        let (universe, _) =
            mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
        let comm = universe.world();
        let mut problems = vec![("C3.2 cube", Config::C32)];
        if kind != BackendKind::Cpu {
            problems.push(("Plummer sphere", Config::c33(Distribution::Plummer)));
        }
        let mut out = Vec::new();
        for &problem in &problems {
            stages::<f32>(kind, problem, arguments, &comm, &mut out);
            if kind == BackendKind::Cpu {
                stages::<f64>(kind, problem, arguments, &comm, &mut out);
            }
        }
        println!("## The leaf stage inside the FMM");
        println!();
        println!(
            "p = 3, gradients, N = 10^5 (the C3.2 cube: uniform level-4 tree; the Plummer \
             sphere: max_level 16, 64 points per leaf). Device runs use synchronous stages, \
             so the leaf stage (L2P, M2P and P2P) is timed whole; L2P and M2P run on the \
             host fallback with their transfers. The median of {} evaluations after a \
             warm-up. Difference: relative L2 of the device output from the host output, φ \
             and ∇φ.",
            arguments.repeats
        );
        println!();
        println!(
            "| problem | precision | configuration | leaf stage ms | evaluation ms | leaf \
             stage x host 1 thread | x host {} threads | difference φ / ∇φ |",
            arguments.threads
        );
        println!("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |");
        for s in &out {
            let host = |label: &str| {
                out.iter()
                    .find(|h| {
                        h.problem == s.problem
                            && h.precision == s.precision
                            && h.configuration == label
                    })
                    .map_or(f64::NAN, |h| h.leaf_stage)
            };
            let one = host("host, 1 thread");
            let many = host(&format!("host, {} threads", arguments.threads));
            println!(
                "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {} |",
                s.problem,
                s.precision,
                s.configuration,
                1e3 * s.leaf_stage,
                1e3 * s.evaluation,
                one / s.leaf_stage,
                many / s.leaf_stage,
                s.difference
                    .map_or("–".into(), |(p, g)| format!("{p:.1e} / {g:.1e}"))
            );
        }
        println!();
    }
}

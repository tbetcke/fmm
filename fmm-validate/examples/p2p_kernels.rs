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
//! Nothing here initialises MPI, although the crate links it. Timings are reported,
//! never asserted. The example builds and runs on x86_64 as well, where it times the
//! AVX2 path; Phase 3S took no x86_64 timings.

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
}

/// Parses `--quick` and `--ghz f`; exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!("usage: p2p_kernels [--quick] [--ghz f], f > 0; got {args:?}");
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        quick: false,
        ghz: cpu_model().contains("Apple M3 Max").then_some(M3_MAX_GHZ),
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
            _ => usage(),
        };
    }
    parsed
}

fn main() {
    let Arguments { quick, ghz } = arguments();
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

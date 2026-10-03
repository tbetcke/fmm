//! The production P2P kernel of `nd-fmm-simd` against the Laplace kernels of
//! green-kernels on NEON (Phase 3S T7, C3S.6). Prints Markdown on stdout and progress
//! on stderr.
//!
//! The inputs are those of `nd-fmm-validate`'s `p2p_kernels` example
//! (`nd_fmm_validate::p2p_kernels`): the W1 (FMM-shaped, per-pair and gathered) and W2
//! (all-pairs) cells of docs/design/simd-p2p.md §8.2, in f32 and f64, with the
//! potential only and with gradients. On each cell:
//!
//! - green-kernels' `Laplace3dKernel::evaluate_st`, `Value` or `ValueDeriv`, one
//!   thread, at the commit pinned in the root `Cargo.toml`, through the pulp backend
//!   printed in the header;
//! - `nd_fmm_simd::P2pKernel::detect()`, the default, then every other available ISA.
//!
//! Both are measured against `direct_sum` on the same four sets, and the reference
//! `nd_fmm_ref::p2p` too (untimed), for the check of requirement 2. green-kernels
//! takes the same interleaved triples, applies 1/(4π) and writes [φ, ∂x, ∂y, ∂z] per
//! target; its output is multiplied by 4π and split before it is measured, and our
//! gradients stay in their own layout. Neither conversion is timed (design §8.3).
//!
//! The C3S.6 target: on NEON the default kernel reaches at least green-kernels' pairs
//! per second, at equal or better accuracy, in every cell. The summary lists the cells
//! where it does not, for the analysis of the T7 report.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-spike-p2p-simd --example compare [-- --quick]
//! ```
//!
//! Timings are reported, never asserted. The example builds and runs on x86_64 too;
//! Phase 3S took no x86_64 timings.

use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as _;
use green_kernels::types::GreenKernelEvalType;
use nd_fmm_simd::{Isa, P2pKernel, SimdScalar};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::p2p_kernels::{
    Accuracy, CHECKED_SETS, Cell, Form, Kernel, Oracle, Outputs, Set, accuracy, check, oracles,
    pairs_per_second, passes, precision_name, time_kernel,
};

/// green-kernels' commit (root `Cargo.toml`).
const GK_COMMIT: &str = "7d757c579f8633d58163cd5b2d011a9468541f17";

/// The precisions green-kernels runs in.
trait Green: SimdScalar {
    /// `Laplace3dKernel::<Self>::evaluate_st`, adding into `out`: one value per target
    /// (`Value`), or φ, ∂x, ∂y, ∂z per target (`ValueDeriv`), scaled by 1/(4π).
    fn green(
        gradients: bool,
        sources: &[[Self; 3]],
        charges: &[Self],
        targets: &[[Self; 3]],
        out: &mut [Self],
    );
}

/// green-kernels' evaluation type for an output.
fn eval_type(gradients: bool) -> GreenKernelEvalType {
    if gradients {
        GreenKernelEvalType::ValueDeriv
    } else {
        GreenKernelEvalType::Value
    }
}

impl Green for f32 {
    fn green(gradients: bool, s: &[[f32; 3]], q: &[f32], t: &[[f32; 3]], out: &mut [f32]) {
        Laplace3dKernel::<f32>::new().evaluate_st(
            eval_type(gradients),
            s.as_flattened(),
            t.as_flattened(),
            q,
            out,
        );
    }
}

impl Green for f64 {
    fn green(gradients: bool, s: &[[f64; 3]], q: &[f64], t: &[[f64; 3]], out: &mut [f64]) {
        Laplace3dKernel::<f64>::new().evaluate_st(
            eval_type(gradients),
            s.as_flattened(),
            t.as_flattened(),
            q,
            out,
        );
    }
}

/// One evaluation of `set` by green-kernels in `form`, adding into `out` (n_t or
/// 4 n_t values).
fn green_evaluate<T: Green>(set: &Set<T>, form: Form, gradients: bool, out: &mut [T]) {
    match form {
        Form::Gathered => T::green(gradients, &set.sources, &set.charges, &set.targets, out),
        Form::PerPair => {
            for leaf in &set.leaves {
                T::green(gradients, &leaf.sources, &leaf.charges, &set.targets, out);
            }
        }
    }
}

/// green-kernels' output in our form, in f64, with its 1/(4π) removed.
fn unpack<T: SimdScalar>(gradients: bool, out: &[T]) -> (Vec<f64>, Option<Vec<[f64; 3]>>) {
    let scale = 4.0 * std::f64::consts::PI;
    let up = |v: T| nd_fmm_math::RealScalar::to_f64(v) * scale;
    if gradients {
        let (values, rest) = out.as_chunks::<4>();
        assert!(rest.is_empty(), "four values per target");
        let potential = values.iter().map(|v| up(v[0])).collect();
        let gradient = values
            .iter()
            .map(|v| [up(v[1]), up(v[2]), up(v[3])])
            .collect();
        (potential, Some(gradient))
    } else {
        (out.iter().map(|&v| up(v)).collect(), None)
    }
}

/// green-kernels' accuracy on the sets of `oracles`, from zeroed outputs.
fn green_check<T: Green>(
    pool: &[Set<T>],
    oracles: &[Oracle],
    form: Form,
    gradients: bool,
) -> Accuracy {
    let width = if gradients { 4 } else { 1 };
    pool.iter()
        .zip(oracles)
        .map(|(set, oracle)| {
            let mut out = vec![T::zero(); width * set.targets.len()];
            green_evaluate(set, form, gradients, &mut out);
            let (potential, gradient) = unpack(gradients, &out);
            accuracy::<f64>(oracle, &potential, gradient.as_deref())
        })
        .fold(Accuracy::default(), Accuracy::worst)
}

/// One kernel of ours on one cell, against green-kernels on the same cell.
struct Row {
    cell: Cell,
    precision: &'static str,
    gradients: bool,
    /// Our ISA.
    isa: Isa,
    /// Whether it is the default kernel, `P2pKernel::detect()`.
    default: bool,
    ours: f64,
    green: f64,
    ours_accuracy: Accuracy,
    green_accuracy: Accuracy,
    /// Requirement 2 (design §3, as decided in T5), against the reference's error.
    ours_passes: bool,
    green_passes: bool,
}

/// Measures green-kernels and every kernel of ours, the default first, on `cell` in
/// precision `T`, with and without gradients. Without `time` the pairs per second are
/// NaN.
fn measure_cell<T: Green>(cell: Cell, time: bool) -> Vec<Row> {
    let pool = cell.pool::<T>();
    let oracles = oracles(&pool);
    let form = cell.form();
    let detected = Isa::detect();
    let kernels: Vec<Isa> = std::iter::once(detected)
        .chain(Isa::available().filter(|&isa| isa != detected))
        .collect();
    let mut rows = Vec::new();
    for gradients in [false, true] {
        eprintln!(
            "{} {} {}",
            precision_name::<T>(),
            cell.label(),
            if gradients { "φ, ∇φ" } else { "φ" }
        );
        let reference = check(&Kernel::Reference, &pool, &oracles, form, gradients);
        let green_accuracy = green_check(&pool, &oracles, form, gradients);
        let green = if time {
            let width = if gradients { 4 } else { 1 };
            let mut outs: Vec<Vec<T>> = pool
                .iter()
                .map(|s| vec![T::zero(); width * s.targets.len()])
                .collect();
            pairs_per_second(pool[0].pairs(), pool.len(), |i| {
                green_evaluate(&pool[i], form, gradients, &mut outs[i]);
            })
        } else {
            f64::NAN
        };
        for &isa in &kernels {
            let kernel = Kernel::Simd(P2pKernel::<T>::new(isa).expect("an available ISA"));
            let ours_accuracy = check(&kernel, &pool, &oracles, form, gradients);
            let ours = if time {
                let mut outputs = Outputs::for_pool(&pool);
                time_kernel(&kernel, &pool, &mut outputs, form, gradients)
            } else {
                f64::NAN
            };
            rows.push(Row {
                cell,
                precision: precision_name::<T>(),
                gradients,
                isa,
                default: isa == detected,
                ours,
                green,
                ours_accuracy,
                green_accuracy,
                ours_passes: passes::<T>(ours_accuracy, reference),
                green_passes: passes::<T>(green_accuracy, reference),
            });
        }
    }
    rows
}

fn main() {
    let quick = match std::env::args().skip(1).collect::<Vec<_>>().as_slice() {
        [] => false,
        [flag] if flag == "--quick" => true,
        args => {
            eprintln!("usage: compare [--quick]; got {args:?}");
            std::process::exit(2);
        }
    };
    let cells = if quick { Cell::quick() } else { Cell::all() };
    let mut rows = Vec::new();
    for &cell in &cells {
        rows.extend(measure_cell::<f32>(cell, true));
    }
    for &cell in &cells {
        rows.extend(measure_cell::<f64>(cell, true));
    }
    print_header(quick);
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            print_table(&rows, precision, gradients);
        }
    }
    print_target(&rows);
}

/// The machine, the toolchain, the backends and the method.
fn print_header(quick: bool) {
    println!("# P2P: nd-fmm-simd against green-kernels (Phase 3S T7)");
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
        "| nd-fmm-simd | default `P2pKernel::detect()` = {}; ISAs available: {} |",
        Isa::detect(),
        Isa::available()
            .map(|isa| isa.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "| green-kernels | commit `{GK_COMMIT}`, `Laplace3dKernel::evaluate_st`, pulp backend `{:?}` |",
        pulp::Arch::new()
    );
    println!(
        "| thread variables | {} |",
        [
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "RAYON_NUM_THREADS",
        ]
        .iter()
        .map(|k| format!(
            "{k}={}",
            std::env::var(k).unwrap_or_else(|_| "unset".into())
        ))
        .collect::<Vec<_>>()
        .join(", ")
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
        "- Inputs: those of `p2p_kernels` (`nd_fmm_validate::p2p_kernels`, design §8.2). \
         Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. \
         \"time ours/gk\": our time per evaluation over green-kernels' (below 1: ours is \
         faster)."
    );
    println!(
        "- Accuracy against `direct_sum` on the first {CHECKED_SETS} sets of the cell, \
         green-kernels' output times 4π: \"max\" is the largest error relative to the \
         term magnitudes (Σ|q|/r for φ, Σ|q|/r² per component of ∇φ), \"L2\" the \
         relative L2 error. \"req. 2\": within 1e-6 (f32) or 1e-14 (f64), or twice the \
         error of `nd_fmm_ref::p2p` on the same inputs (design §3, as decided in T5)."
    );
    println!(
        "- Target (C3S.6): speed, ours ≥ green-kernels' Gpairs/s; accuracy, our largest \
         error (max of φ and ∇φ) ≤ green-kernels'."
    );
    println!();
}

/// "ours / theirs" of the max errors, and the two max errors.
fn errors(ours: f64, green: f64) -> String {
    format!("{ours:.1e} / {green:.1e}")
}

/// One table per precision and output: a row per cell and kernel of ours.
fn print_table(rows: &[Row], precision: &str, gradients: bool) {
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
        "| cell | ours | ours Gpairs/s | green-kernels Gpairs/s | time ours/gk | max φ ours / gk | L2 φ ours / gk | max ∇φ ours / gk | L2 ∇φ ours / gk | req. 2 ours / gk | speed | accuracy |"
    );
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |");
    for r in rows
        .iter()
        .filter(|r| r.precision == precision && r.gradients == gradients)
    {
        let (o, g) = (r.ours_accuracy, r.green_accuracy);
        let ok = |b: bool| if b { "pass" } else { "FAIL" };
        println!(
            "| {} | {}{} | {:.3} | {:.3} | {:.2} | {} | {} | {} | {} | {} / {} | {} | {} |",
            r.cell.label(),
            r.isa,
            if r.default { " (default)" } else { "" },
            r.ours / 1e9,
            r.green / 1e9,
            r.green / r.ours,
            errors(o.max_potential, g.max_potential),
            errors(o.l2_potential, g.l2_potential),
            if gradients {
                errors(o.max_gradient, g.max_gradient)
            } else {
                String::new()
            },
            if gradients {
                errors(o.l2_gradient, g.l2_gradient)
            } else {
                String::new()
            },
            ok(r.ours_passes),
            ok(r.green_passes),
            if r.ours >= r.green { "met" } else { "BELOW" },
            if o.max() <= g.max() { "met" } else { "worse" },
        );
    }
    println!();
}

/// The geometric mean of positive numbers.
fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// The C3S.6 target for the default kernel: geometric means and the cells below it.
fn print_target(rows: &[Row]) {
    println!("## The C3S.6 target, default kernel");
    println!();
    println!(
        "Speed-up over green-kernels (its time over ours), geometric mean over the cells \
         of each group, its smallest value, and the cells where the target is not met."
    );
    println!();
    println!(
        "| precision | output | W1 per-pair | W1 gathered | W2 | smallest (cell) | speed met | accuracy met | cells below in speed | cells worse in accuracy (ours / gk max error) |"
    );
    println!("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |");
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            let mine: Vec<&Row> = rows
                .iter()
                .filter(|r| r.default && r.precision == precision && r.gradients == gradients)
                .collect();
            let group = |g: fn(&Cell) -> bool| -> String {
                let v: Vec<f64> = mine
                    .iter()
                    .filter(|r| g(&r.cell))
                    .map(|r| r.ours / r.green)
                    .collect();
                if v.is_empty() {
                    "–".into()
                } else {
                    format!("{:.2}", geomean(&v))
                }
            };
            let smallest = mine
                .iter()
                .min_by(|a, b| (a.ours / a.green).total_cmp(&(b.ours / b.green)))
                .map_or("–".into(), |r| {
                    format!("{:.2} ({})", r.ours / r.green, r.cell.label())
                });
            let slow: Vec<String> = mine
                .iter()
                .filter(|r| r.ours < r.green)
                .map(|r| format!("{} ({:.2})", r.cell.label(), r.ours / r.green))
                .collect();
            let worse: Vec<String> = mine
                .iter()
                .filter(|r| r.ours_accuracy.max() > r.green_accuracy.max())
                .map(|r| {
                    format!(
                        "{} ({:.1e} / {:.1e})",
                        r.cell.label(),
                        r.ours_accuracy.max(),
                        r.green_accuracy.max()
                    )
                })
                .collect();
            println!(
                "| {precision} | {} | {} | {} | {} | {smallest} | {}/{} | {}/{} | {} | {} |",
                if gradients { "φ, ∇φ" } else { "φ" },
                group(|c| matches!(
                    c,
                    Cell::W1 {
                        form: Form::PerPair,
                        ..
                    }
                )),
                group(|c| matches!(
                    c,
                    Cell::W1 {
                        form: Form::Gathered,
                        ..
                    }
                )),
                group(|c| matches!(c, Cell::W2 { .. })),
                mine.len() - slow.len(),
                mine.len(),
                mine.len() - worse.len(),
                mine.len(),
                if slow.is_empty() {
                    "none".into()
                } else {
                    slow.join("; ")
                },
                if worse.is_empty() {
                    "none".into()
                } else {
                    worse.join("; ")
                },
            );
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core on the smallest W1 cell, both forms: green-kernels and every kernel of
    /// ours within requirement 2 of `direct_sum`; timed once in f64.
    #[test]
    fn compare_core_on_a_small_cell() {
        println!(
            "ISAs: {:?}, default {}, pulp backend {:?}",
            Isa::available().collect::<Vec<_>>(),
            Isa::detect(),
            pulp::Arch::new()
        );
        for form in [Form::Gathered, Form::PerPair] {
            let cell = Cell::W1 { targets: 8, form };
            let mut rows = measure_cell::<f32>(cell, false);
            rows.extend(measure_cell::<f64>(cell, form == Form::Gathered));
            assert_eq!(rows.len(), 4 * Isa::available().count());
            for r in &rows {
                println!(
                    "{} {} {} gradients={}: ours {:?}, gk {:?}",
                    r.precision,
                    r.cell.label(),
                    r.isa,
                    r.gradients,
                    r.ours_accuracy,
                    r.green_accuracy
                );
                assert!(r.ours_passes && r.green_passes);
                assert!(r.default == (r.isa == Isa::detect()));
                if r.precision == "f64" && form == Form::Gathered {
                    assert!(r.ours > 0.0 && r.green > 0.0);
                }
            }
        }
        // The unpacking removes 1/(4π) and splits the four values per target.
        let pi4 = 1.0 / (4.0 * std::f64::consts::PI);
        let (p, g) = unpack(true, &[pi4, 2.0 * pi4, 3.0 * pi4, 4.0 * pi4]);
        assert!((p[0] - 1.0).abs() < 1e-15);
        let g = g.expect("gradients");
        assert!((g[0][2] - 4.0).abs() < 1e-15);
    }
}

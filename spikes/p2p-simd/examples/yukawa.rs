//! The cost per pair of the Yukawa P2P (Helmholtz CH2.4b) against Laplace's, for
//! `nd-fmm-simd` and for green-kernels' modified Helmholtz kernel. Prints Markdown on
//! stdout and progress on stderr.
//!
//! The inputs are the W1 cells of `nd-fmm-validate`'s `p2p_kernels` example
//! (`nd_fmm_validate::p2p_kernels`, docs/design/simd-p2p.md §8.2): a target leaf of n_t
//! points in [−1, 1]³ (leaf-scaled, leaf width 2) and the 27 leaves of its 3 × 3 × 3
//! block, per-pair (27 calls) and gathered (one call), in f32 and f64, with the potential
//! only and with gradients. κ is in the same leaf-scaled units (`--kappa`, default 1, so
//! κ times the leaf width is 2). On each cell, one thread:
//!
//! - `P2pKernel::evaluate_yukawa` and `P2pKernel::evaluate` on every available ISA, the
//!   default first;
//! - green-kernels' `ModifiedHelmholtz3dKernel::evaluate_st` (ω = κ) and
//!   `Laplace3dKernel::evaluate_st` at the commit pinned in the root `Cargo.toml`, through
//!   the pulp backend printed in the header.
//!
//! "Yukawa / Laplace" is the time per pair of the Yukawa kernel over the Laplace kernel of
//! the same code on the same cell. The Yukawa kernels are measured against
//! `nd_fmm_ref::helmholtz::p2p::direct_sum_yukawa` on the first four sets of the cell, and
//! so is the reference `p2p_yukawa` (untimed). green-kernels' output is multiplied by 4π
//! and split as in `compare`; its gradient is ∂/∂x like ours, which the example checks
//! on one pair before it measures (the convention check of proposal §5.7). The
//! exponential has no branch, so the times do not depend on κ.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-spike-p2p-simd --example yukawa [-- --quick] [--kappa κ]
//! ```
//!
//! Timings are reported, never asserted.

use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::modified_helmholtz_3d::ModifiedHelmholtz3dKernel;
use green_kernels::traits::Kernel as _;
use green_kernels::types::GreenKernelEvalType;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::helmholtz::p2p::{direct_sum_yukawa, p2p_yukawa};
use nd_fmm_simd::{Isa, P2pKernel, SimdScalar};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::p2p_kernels::{
    CHECKED_SETS, Cell, Form, Set, W1_TARGETS, pairs_per_second, precision_name,
};

/// green-kernels' commit (root `Cargo.toml`).
const GK_COMMIT: &str = "7d757c579f8633d58163cd5b2d011a9468541f17";

/// The precisions green-kernels runs in.
trait Green: SimdScalar {
    /// `ModifiedHelmholtz3dKernel::<Self>::new(omega)` (or, for ω = `None`,
    /// `Laplace3dKernel`) `evaluate_st`, adding into `out`: one value per target, or
    /// φ, ∂x, ∂y, ∂z per target, scaled by 1/(4π).
    fn green(
        omega: Option<Self>,
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

macro_rules! green_impl {
    ($t:ty) => {
        impl Green for $t {
            fn green(
                omega: Option<$t>,
                gradients: bool,
                s: &[[$t; 3]],
                q: &[$t],
                t: &[[$t; 3]],
                out: &mut [$t],
            ) {
                let (s, t, e) = (s.as_flattened(), t.as_flattened(), eval_type(gradients));
                match omega {
                    Some(omega) => {
                        ModifiedHelmholtz3dKernel::<$t>::new(omega).evaluate_st(e, s, t, q, out)
                    }
                    None => Laplace3dKernel::<$t>::new().evaluate_st(e, s, t, q, out),
                }
            }
        }
    };
}

green_impl!(f32);
green_impl!(f64);

/// The kernels timed on a cell.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Code {
    /// `nd-fmm-simd` on an ISA.
    Ours(Isa),
    /// green-kernels.
    Green,
}

impl Code {
    fn name(self) -> String {
        match self {
            Code::Ours(isa) if isa == Isa::detect() => format!("nd-fmm-simd {isa} (default)"),
            Code::Ours(isa) => format!("nd-fmm-simd {isa}"),
            Code::Green => "green-kernels".into(),
        }
    }
}

/// One evaluation of `set` by `code` in `form`, Yukawa (`kappa`) or Laplace (`None`),
/// adding into `potential` and `gradient` (ours) or `out` (green-kernels, n_t or 4 n_t
/// values).
#[expect(
    clippy::too_many_arguments,
    reason = "one call site per kernel, both layouts"
)]
fn evaluate<T: Green>(
    code: Code,
    kappa: Option<T>,
    set: &Set<T>,
    form: Form,
    gradients: bool,
    potential: &mut [T],
    gradient: &mut [[T; 3]],
    out: &mut [T],
) {
    let calls: Vec<(&[[T; 3]], &[T])> = match form {
        Form::Gathered => vec![(&set.sources, &set.charges)],
        Form::PerPair => set
            .leaves
            .iter()
            .map(|l| (&l.sources[..], &l.charges[..]))
            .collect(),
    };
    for (sources, charges) in calls {
        match code {
            Code::Ours(isa) => {
                let kernel = P2pKernel::<T>::new(isa).expect("an available ISA");
                let g = gradients.then_some(&mut *gradient);
                match kappa {
                    Some(kappa) => {
                        kernel.evaluate_yukawa(kappa, sources, charges, &set.targets, potential, g)
                    }
                    None => kernel.evaluate(sources, charges, &set.targets, potential, g),
                }
            }
            Code::Green => T::green(kappa, gradients, sources, charges, &set.targets, out),
        }
    }
}

/// The largest errors of a Yukawa result against `direct_sum_yukawa` on one set, relative
/// per target to Σ |q| e^{−κr}/r (potential) and Σ |q| (1 + κr) e^{−κr}/r² (each gradient
/// component), over the pairs with r > 0.
fn errors<T: SimdScalar>(
    kappa: f64,
    set: &Set<T>,
    potential: &[f64],
    gradient: Option<&[[f64; 3]]>,
) -> (f64, f64) {
    let widen =
        |p: &[[T; 3]]| -> Vec<[f64; 3]> { p.iter().map(|x| x.map(RealScalar::to_f64)).collect() };
    let (sources, targets) = (widen(&set.sources), widen(&set.targets));
    let charges: Vec<f64> = set.charges.iter().map(|&q| q.to_f64()).collect();
    let mut want_p = vec![0.0; targets.len()];
    let mut want_g = vec![[0.0; 3]; targets.len()];
    direct_sum_yukawa(
        kappa,
        &sources,
        &charges,
        &targets,
        &mut want_p,
        Some(&mut want_g),
    );
    let (mut ep, mut eg) = (0.0f64, 0.0f64);
    for (i, x) in targets.iter().enumerate() {
        let (mut sp, mut sg) = (0.0, 0.0);
        for (y, q) in sources.iter().zip(&charges) {
            let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
            if r2 > 0.0 {
                let r = r2.sqrt();
                let decay = (-kappa * r).exp();
                sp += q.abs() * decay / r;
                sg += q.abs() * (1.0 + kappa * r) * decay / r2;
            }
        }
        ep = ep.max((potential[i] - want_p[i]).abs() / sp);
        if let Some(g) = gradient {
            for k in 0..3 {
                eg = eg.max((g[i][k] - want_g[i][k]).abs() / sg);
            }
        }
    }
    (ep, eg)
}

/// green-kernels' output in our form, in f64, with its 1/(4π) removed.
fn unpack<T: SimdScalar>(gradients: bool, out: &[T]) -> (Vec<f64>, Option<Vec<[f64; 3]>>) {
    let scale = 4.0 * std::f64::consts::PI;
    let up = |v: T| v.to_f64() * scale;
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

/// The largest errors of `code`'s Yukawa kernel (or, for `None`, `p2p_yukawa`) on the first
/// [`CHECKED_SETS`] sets of `pool`.
fn accuracy<T: Green>(
    code: Option<Code>,
    kappa: T,
    pool: &[Set<T>],
    form: Form,
    gradients: bool,
) -> (f64, f64) {
    let mut worst = (0.0f64, 0.0f64);
    for set in pool.iter().take(CHECKED_SETS) {
        let n = set.targets.len();
        let (mut p, mut g, mut out) = (
            vec![T::zero(); n],
            vec![[T::zero(); 3]; n],
            vec![T::zero(); 4 * n],
        );
        let (potential, gradient): (Vec<f64>, Option<Vec<[f64; 3]>>) = match code {
            Some(Code::Green) => {
                let out = &mut out[..if gradients { 4 * n } else { n }];
                evaluate(
                    Code::Green,
                    Some(kappa),
                    set,
                    form,
                    gradients,
                    &mut p,
                    &mut g,
                    out,
                );
                unpack(gradients, out)
            }
            Some(code) => {
                evaluate(
                    code,
                    Some(kappa),
                    set,
                    form,
                    gradients,
                    &mut p,
                    &mut g,
                    &mut out,
                );
                let widen = |v: &[T]| v.iter().map(|&x| RealScalar::to_f64(x)).collect::<Vec<_>>();
                let gw = g.iter().map(|x| x.map(|c| c.to_f64())).collect();
                (widen(&p), gradients.then_some(gw))
            }
            None => {
                p2p_yukawa(
                    kappa,
                    &set.sources,
                    &set.charges,
                    &set.targets,
                    &mut p,
                    gradients.then_some(&mut g[..]),
                );
                let gw = g.iter().map(|x| x.map(|c| c.to_f64())).collect();
                (
                    p.iter().map(|&x| RealScalar::to_f64(x)).collect(),
                    gradients.then_some(gw),
                )
            }
        };
        let e = errors(kappa.to_f64(), set, &potential, gradient.as_deref());
        worst = (worst.0.max(e.0), worst.1.max(e.1));
    }
    worst
}

/// One code on one cell and output: pairs per second of the Yukawa and the Laplace
/// kernel, and the Yukawa kernel's accuracy.
struct Row {
    cell: Cell,
    precision: &'static str,
    gradients: bool,
    code: Code,
    yukawa: f64,
    laplace: f64,
    errors: (f64, f64),
    reference: (f64, f64),
}

/// The outputs of one set: our potential and gradient, and green-kernels' values.
type SetOutputs<T> = (Vec<T>, Vec<[T; 3]>, Vec<T>);

/// The pairs per second of `code` on `pool`, Yukawa (`kappa`) or Laplace (`None`).
fn time<T: Green>(code: Code, kappa: Option<T>, pool: &[Set<T>], form: Form, g: bool) -> f64 {
    let n = pool[0].targets.len();
    let mut outs: Vec<SetOutputs<T>> = pool
        .iter()
        .map(|_| {
            (
                vec![T::zero(); n],
                vec![[T::zero(); 3]; n],
                vec![T::zero(); if g { 4 * n } else { n }],
            )
        })
        .collect();
    pairs_per_second(pool[0].pairs(), pool.len(), |i| {
        let (p, gr, out) = &mut outs[i];
        evaluate(code, kappa, &pool[i], form, g, p, gr, out);
    })
}

fn measure_cell<T: Green>(cell: Cell, kappa: T) -> Vec<Row> {
    let pool = cell.pool::<T>();
    let form = cell.form();
    let detected = Isa::detect();
    let codes: Vec<Code> = std::iter::once(detected)
        .chain(Isa::available().filter(|&isa| isa != detected))
        .map(Code::Ours)
        .chain([Code::Green])
        .collect();
    let mut rows = Vec::new();
    for gradients in [false, true] {
        eprintln!(
            "{} {} {}",
            precision_name::<T>(),
            cell.label(),
            if gradients { "φ, ∇φ" } else { "φ" }
        );
        let reference = accuracy(None, kappa, &pool, form, gradients);
        for &code in &codes {
            rows.push(Row {
                cell,
                precision: precision_name::<T>(),
                gradients,
                code,
                yukawa: time(code, Some(kappa), &pool, form, gradients),
                laplace: time(code, None, &pool, form, gradients),
                errors: accuracy(Some(code), kappa, &pool, form, gradients),
                reference,
            });
        }
    }
    rows
}

/// The convention check: green-kernels' modified Helmholtz kernel on one pair, times 4π,
/// against `direct_sum_yukawa` (potential and ∂/∂x), in f64. Panics on a mismatch.
fn convention_check(kappa: f64) -> String {
    let (sources, charges, targets) = ([[0.1, -0.2, 0.3]], [0.7], [[0.9, 0.4, -0.5]]);
    let mut want_p = [0.0];
    let mut want_g = [[0.0; 3]];
    direct_sum_yukawa(
        kappa,
        &sources,
        &charges,
        &targets,
        &mut want_p,
        Some(&mut want_g),
    );
    let mut out = [0.0; 4];
    f64::green(Some(kappa), true, &sources, &charges, &targets, &mut out);
    let (p, g) = unpack(true, &out);
    let g = g.expect("gradients");
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    let worst = (0..3)
        .map(|k| rel(g[0][k], want_g[0][k]))
        .fold(rel(p[0], want_p[0]), f64::max);
    assert!(
        worst < 1e-10,
        "green-kernels' modified Helmholtz convention differs: {p:?} {g:?} against \
         {want_p:?} {want_g:?}"
    );
    format!(
        "one pair at κ = {kappa}: 4π × green-kernels' φ and ∂/∂x equal `direct_sum_yukawa` \
         within {worst:.1e} relative (factor 1/(4π), gradient with respect to the target, \
         e^{{−ωr}}/r with ω = κ)"
    )
}

fn main() {
    let mut quick = false;
    let mut kappa = 1.0f64;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rest = args.as_slice();
    while let Some((flag, tail)) = rest.split_first() {
        rest = match (flag.as_str(), tail) {
            ("--quick", _) => {
                quick = true;
                tail
            }
            ("--kappa", [value, tail @ ..]) => {
                kappa = value.parse().expect("--kappa takes a number");
                tail
            }
            _ => {
                eprintln!("usage: yukawa [--quick] [--kappa κ]; got {args:?}");
                std::process::exit(2);
            }
        };
    }
    let convention = convention_check(kappa);
    let targets: &[usize] = if quick { &[24] } else { &W1_TARGETS };
    let cells: Vec<Cell> = targets
        .iter()
        .flat_map(|&n| [Form::PerPair, Form::Gathered].map(|form| Cell::W1 { targets: n, form }))
        .collect();
    let mut rows = Vec::new();
    for &cell in &cells {
        rows.extend(measure_cell::<f32>(cell, kappa as f32));
    }
    for &cell in &cells {
        rows.extend(measure_cell::<f64>(cell, kappa));
    }
    print_header(quick, kappa, &convention);
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            print_table(&rows, precision, gradients);
        }
    }
    print_summary(&rows);
}

/// The machine, the toolchain, the backends and the method.
fn print_header(quick: bool, kappa: f64, convention: &str) {
    println!("# Yukawa P2P: cost per pair against Laplace (Helmholtz CH2.4b)");
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
        "| green-kernels | commit `{GK_COMMIT}`, `ModifiedHelmholtz3dKernel` and \
         `Laplace3dKernel::evaluate_st`, pulp backend `{:?}` |",
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
        "| mode | {}, κ = {kappa} (leaf-scaled, leaf width 2) |",
        if quick {
            "--quick (W1 n_t = 24)"
        } else {
            "full"
        }
    );
    println!("| timing | median of 15 batches of ≥ 20 ms, one thread |");
    println!("| convention check | {convention} |");
    println!();
    println!(
        "- Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. \
         \"Yukawa / Laplace\": the Yukawa kernel's time per pair over the Laplace kernel's, \
         same code, same cell."
    );
    println!(
        "- Accuracy of the Yukawa kernel against `direct_sum_yukawa` on the first \
         {CHECKED_SETS} sets: the largest error relative to Σ|q|e^{{−κr}}/r (φ) and \
         Σ|q|(1+κr)e^{{−κr}}/r² (per component of ∇φ); \"ref\" is `p2p_yukawa`'s."
    );
    println!();
}

/// One table per precision and output: a row per cell and code.
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
        "| cell | code | Yukawa Gpairs/s | Laplace Gpairs/s | Yukawa / Laplace | max φ | max ∇φ | ref max φ | ref max ∇φ |"
    );
    println!("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for r in rows
        .iter()
        .filter(|r| r.precision == precision && r.gradients == gradients)
    {
        let grad = |e: f64| {
            if gradients {
                format!("{e:.1e}")
            } else {
                String::new()
            }
        };
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.2} | {:.1e} | {} | {:.1e} | {} |",
            r.cell.label(),
            r.code.name(),
            r.yukawa / 1e9,
            r.laplace / 1e9,
            r.laplace / r.yukawa,
            r.errors.0,
            grad(r.errors.1),
            r.reference.0,
            grad(r.reference.1),
        );
    }
    println!();
}

/// The geometric mean of positive numbers.
fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// Geometric means over the cells of each form: Yukawa / Laplace per code, and our
/// default Yukawa kernel's speed over green-kernels'.
fn print_summary(rows: &[Row]) {
    println!("## Summary");
    println!();
    println!(
        "Geometric means over the W1 cells of each form. \"Yukawa / Laplace\" per code; \
         \"ours / gk\": the default kernel's Yukawa pairs per second over green-kernels' \
         modified Helmholtz (above 1: ours is faster)."
    );
    println!();
    println!(
        "| precision | output | form | default Yukawa Gpairs/s | default Yukawa / Laplace | gk Yukawa / Laplace | ours / gk (Yukawa) | ours / gk (Laplace) |"
    );
    println!("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |");
    let default = Code::Ours(Isa::detect());
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            for form in [Form::PerPair, Form::Gathered] {
                let pick = |code: Code| -> Vec<&Row> {
                    rows.iter()
                        .filter(|r| {
                            r.precision == precision
                                && r.gradients == gradients
                                && r.code == code
                                && r.cell.form() == form
                        })
                        .collect()
                };
                let (ours, gk) = (pick(default), pick(Code::Green));
                let mean = |v: &[&Row], f: fn(&Row) -> f64| {
                    geomean(&v.iter().map(|r| f(r)).collect::<Vec<_>>())
                };
                let ratio = |a: &[&Row], b: &[&Row], f: fn(&Row) -> f64| {
                    geomean(
                        &a.iter()
                            .zip(b)
                            .map(|(x, y)| f(x) / f(y))
                            .collect::<Vec<_>>(),
                    )
                };
                println!(
                    "| {precision} | {} | {} | {:.3} | {:.2} | {:.2} | {:.2} | {:.2} |",
                    if gradients { "φ, ∇φ" } else { "φ" },
                    form.name(),
                    mean(&ours, |r| r.yukawa) / 1e9,
                    mean(&ours, |r| r.laplace / r.yukawa),
                    mean(&gk, |r| r.laplace / r.yukawa),
                    ratio(&ours, &gk, |r| r.yukawa),
                    ratio(&ours, &gk, |r| r.laplace),
                );
            }
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core on the smallest W1 cell, gathered, in f64: the convention check passes,
    /// every Yukawa kernel of ours meets twice the reference's error or 1e-14, and every
    /// time is positive.
    #[test]
    fn yukawa_core_on_a_small_cell() {
        println!("{}", convention_check(1.0));
        let cell = Cell::W1 {
            targets: 8,
            form: Form::Gathered,
        };
        for r in measure_cell::<f64>(cell, 1.0) {
            println!(
                "{} gradients={}: errors {:?}, reference {:?}",
                r.code.name(),
                r.gradients,
                r.errors,
                r.reference
            );
            assert!(r.yukawa > 0.0 && r.laplace > 0.0);
            if let Code::Ours(_) = r.code {
                assert!(r.errors.0 <= 1e-14_f64.max(2.0 * r.reference.0));
                assert!(r.errors.1 <= 1e-14_f64.max(2.0 * r.reference.1));
            }
        }
    }
}

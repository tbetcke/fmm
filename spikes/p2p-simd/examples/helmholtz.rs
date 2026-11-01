//! The cost per pair of the Helmholtz P2P (Helmholtz CH3.4b) against Laplace's and Yukawa's,
//! for `nd-fmm-simd` and for green-kernels' Helmholtz kernel. Prints Markdown on stdout and
//! progress on stderr.
//!
//! The inputs are the W1 cells of `nd-fmm-validate`'s `p2p_kernels` example
//! (`nd_fmm_validate::p2p_kernels`, docs/design/simd-p2p.md §8.2): a target leaf of n_t
//! points in [−1, 1]³ (leaf-scaled, leaf width 2) and the 27 leaves of its 3 × 3 × 3
//! block, per-pair (27 calls) and gathered (one call), in f32 and f64, with the potential
//! only and with gradients. k is in the same leaf-scaled units (`--k`, default 1, so Re(k)
//! times the leaf width is 2; `--im`, default 0.1, the damping of the damped variant). On
//! each cell, one thread:
//!
//! - `P2pKernel::evaluate_helmholtz` on every available ISA, the default first, in three
//!   variants: real charges and real k ("real q"), complex charges and real k ("complex
//!   q"), complex charges and k = Re k + i Im k ("damped"); beside `evaluate_yukawa`
//!   (κ = Re k) and `evaluate` of the same ISA;
//! - green-kernels' `Helmholtz3dKernel::evaluate_st` (complex charges, real k) and
//!   `Laplace3dKernel::evaluate_st` at the commit pinned in the root `Cargo.toml`, through the
//!   pulp backend printed in the header.
//!
//! The complex charges are the cell's charges plus i times the same charges in reverse
//! order. "complex q / Laplace" and "complex q / Yukawa" are the time per pair of the
//! complex-charge Helmholtz kernel over the Laplace and Yukawa kernels of the same code on
//! the same cell. The complex-charge kernels are measured against
//! `nd_fmm_ref::helmholtz::p2p::direct_sum` on the first four sets of the cell, and so is
//! the reference `p2p` (untimed). green-kernels' output is multiplied by 4π; its gradient
//! is ∂/∂x like ours and its kernel e^{ikr}/r, which the example checks on one pair before it
//! measures (the convention check of proposal §5.7). The kernels have no branch on the
//! data, so the times do not depend on k.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-spike-p2p-simd --example helmholtz [-- --quick] [--k k] [--im d]
//! ```
//!
//! Timings are reported, never asserted.

use green_kernels::helmholtz_3d::Helmholtz3dKernel;
use green_kernels::laplace_3d::Laplace3dKernel;
use green_kernels::traits::Kernel as _;
use green_kernels::types::GreenKernelEvalType;
use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::helmholtz::p2p::{direct_sum, p2p};
use nd_fmm_simd::{HelmholtzCharges, Isa, P2pKernel, SimdScalar};
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};
use nd_fmm_validate::p2p_kernels::{
    CHECKED_SETS, Cell, Form, Set, W1_TARGETS, pairs_per_second, precision_name,
};

/// green-kernels' commit (root `Cargo.toml`).
const GK_COMMIT: &str = "7d757c579f8633d58163cd5b2d011a9468541f17";

/// The precisions green-kernels runs in.
trait Green: SimdScalar {
    /// `Helmholtz3dKernel::<Complex<Self>>::new(k)` `evaluate_st` with complex charges `q`,
    /// adding into `out`: one value per target, or φ, ∂x, ∂y, ∂z per target, scaled by
    /// 1/(4π).
    fn green_helmholtz(
        k: Self,
        gradients: bool,
        sources: &[[Self; 3]],
        q: &[Complex<Self>],
        targets: &[[Self; 3]],
        out: &mut [Complex<Self>],
    );

    /// `Laplace3dKernel::<Self>::new()` `evaluate_st`, as above with real charges.
    fn green_laplace(
        gradients: bool,
        sources: &[[Self; 3]],
        q: &[Self],
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
            fn green_helmholtz(
                k: $t,
                gradients: bool,
                s: &[[$t; 3]],
                q: &[Complex<$t>],
                t: &[[$t; 3]],
                out: &mut [Complex<$t>],
            ) {
                let (s, t, e) = (s.as_flattened(), t.as_flattened(), eval_type(gradients));
                Helmholtz3dKernel::<Complex<$t>>::new(k).evaluate_st(e, s, t, q, out);
            }

            fn green_laplace(
                gradients: bool,
                s: &[[$t; 3]],
                q: &[$t],
                t: &[[$t; 3]],
                out: &mut [$t],
            ) {
                let (s, t, e) = (s.as_flattened(), t.as_flattened(), eval_type(gradients));
                Laplace3dKernel::<$t>::new().evaluate_st(e, s, t, q, out);
            }
        }
    };
}

green_impl!(f32);
green_impl!(f64);

/// What is timed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Variant {
    /// Helmholtz, real charges, real k.
    RealQ,
    /// Helmholtz, complex charges, real k.
    ComplexQ,
    /// Helmholtz, complex charges, complex k.
    Damped,
    /// Yukawa, κ = Re k.
    Yukawa,
    /// Laplace.
    Laplace,
}

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

/// One call of a set's form: the sources, the real charges, the point-major complex charges
/// (Re q, Im q per source) and the complex charges.
type Call<T> = (Vec<[T; 3]>, Vec<T>, Vec<T>, Vec<Complex<T>>);

/// A set with its complex charges, per call of its form.
struct Calls<T> {
    calls: Vec<Call<T>>,
}

impl<T: SimdScalar> Calls<T> {
    fn new(set: &Set<T>, form: Form) -> Self {
        let parts: Vec<(&[[T; 3]], &[T])> = match form {
            Form::Gathered => vec![(&set.sources, &set.charges)],
            Form::PerPair => set
                .leaves
                .iter()
                .map(|l| (&l.sources[..], &l.charges[..]))
                .collect(),
        };
        let calls = parts
            .into_iter()
            .map(|(sources, charges)| {
                let complex: Vec<Complex<T>> = charges
                    .iter()
                    .zip(charges.iter().rev())
                    .map(|(&re, &im)| Complex::new(re, im))
                    .collect();
                let planes = complex.iter().flat_map(|q| [q.re, q.im]).collect();
                (sources.to_vec(), charges.to_vec(), planes, complex)
            })
            .collect();
        Self { calls }
    }
}

/// The outputs of one set: our planes, green-kernels' complex and real values.
struct Outputs<T> {
    potential: Vec<T>,
    gradient: Vec<[T; 3]>,
    green_complex: Vec<Complex<T>>,
    green_real: Vec<T>,
}

impl<T: SimdScalar> Outputs<T> {
    fn new(n: usize) -> Self {
        Self {
            potential: vec![T::zero(); 2 * n],
            gradient: vec![[T::zero(); 3]; 2 * n],
            green_complex: vec![Complex::new(T::zero(), T::zero()); 4 * n],
            green_real: vec![T::zero(); 4 * n],
        }
    }
}

/// One evaluation of `set` by `code` in `variant`, adding into `out`.
fn evaluate<T: Green>(
    code: Code,
    variant: Variant,
    k: Complex<T>,
    set: &Set<T>,
    calls: &Calls<T>,
    gradients: bool,
    out: &mut Outputs<T>,
) {
    let n = set.targets.len();
    let targets = &set.targets;
    for (sources, real, planes, complex) in &calls.calls {
        match code {
            Code::Ours(isa) => {
                let kernel = P2pKernel::<T>::new(isa).expect("an available ISA");
                let (p, g) = (&mut out.potential, &mut out.gradient);
                let real_k = Complex::new(k.re, T::zero());
                match variant {
                    Variant::RealQ | Variant::ComplexQ | Variant::Damped => {
                        let (charges, k) = match variant {
                            Variant::RealQ => (HelmholtzCharges::Real(real), real_k),
                            Variant::ComplexQ => (HelmholtzCharges::Complex(planes), real_k),
                            _ => (HelmholtzCharges::Complex(planes), k),
                        };
                        let g = gradients.then_some(&mut g[..]);
                        kernel.evaluate_helmholtz(k, sources, charges, targets, p, g);
                    }
                    Variant::Yukawa => {
                        let g = gradients.then_some(&mut g[..n]);
                        kernel.evaluate_yukawa(k.re, sources, real, targets, &mut p[..n], g);
                    }
                    Variant::Laplace => {
                        let g = gradients.then_some(&mut g[..n]);
                        kernel.evaluate(sources, real, targets, &mut p[..n], g);
                    }
                }
            }
            Code::Green => {
                let m = if gradients { 4 * n } else { n };
                match variant {
                    Variant::ComplexQ => T::green_helmholtz(
                        k.re,
                        gradients,
                        sources,
                        complex,
                        targets,
                        &mut out.green_complex[..m],
                    ),
                    Variant::Laplace => {
                        T::green_laplace(
                            gradients,
                            sources,
                            real,
                            targets,
                            &mut out.green_real[..m],
                        );
                    }
                    _ => unreachable!("green-kernels runs complex q and Laplace only"),
                }
            }
        }
    }
}

/// The largest errors of a Helmholtz result (complex charges, real k) against `direct_sum`
/// on one set, relative per target to Σ |q|/r (potential) and Σ |q| (1 + |k| r)/r² (each part
/// of each gradient component), over the pairs with r > 0.
fn errors<T: SimdScalar>(
    k: f64,
    set: &Set<T>,
    complex: &[Complex<T>],
    potential: &[Complex<f64>],
    gradient: Option<&[[Complex<f64>; 3]]>,
) -> (f64, f64) {
    let widen =
        |p: &[[T; 3]]| -> Vec<[f64; 3]> { p.iter().map(|x| x.map(RealScalar::to_f64)).collect() };
    let (sources, targets) = (widen(&set.sources), widen(&set.targets));
    let charges: Vec<Complex<f64>> = complex
        .iter()
        .map(|q| Complex::new(q.re.to_f64(), q.im.to_f64()))
        .collect();
    let mut want_p = vec![Complex::new(0.0, 0.0); targets.len()];
    let mut want_g = vec![[Complex::new(0.0, 0.0); 3]; targets.len()];
    direct_sum(
        Complex::new(k, 0.0),
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
            let r2: f64 = (0..3).map(|c| (x[c] - y[c]).powi(2)).sum();
            if r2 > 0.0 {
                let r = r2.sqrt();
                sp += q.norm() / r;
                sg += q.norm() * (1.0 + k * r) / r2;
            }
        }
        let d = potential[i] - want_p[i];
        ep = ep.max(d.re.abs().max(d.im.abs()) / sp);
        if let Some(g) = gradient {
            for c in 0..3 {
                let d = g[i][c] - want_g[i][c];
                eg = eg.max(d.re.abs().max(d.im.abs()) / sg);
            }
        }
    }
    (ep, eg)
}

/// green-kernels' Helmholtz output in our form, in f64, with its 1/(4π) removed.
type Complex64Outputs = (Vec<Complex<f64>>, Option<Vec<[Complex<f64>; 3]>>);

fn unpack<T: SimdScalar>(gradients: bool, out: &[Complex<T>]) -> Complex64Outputs {
    let scale = 4.0 * std::f64::consts::PI;
    let up = |v: Complex<T>| Complex::new(v.re.to_f64() * scale, v.im.to_f64() * scale);
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

/// Our planes, in f64, as complex values.
fn from_planes<T: SimdScalar>(n: usize, out: &Outputs<T>, gradients: bool) -> Complex64Outputs {
    let w = |v: T| v.to_f64();
    let potential = (0..n)
        .map(|i| Complex::new(w(out.potential[i]), w(out.potential[n + i])))
        .collect();
    let gradient = gradients.then(|| {
        (0..n)
            .map(|i| {
                core::array::from_fn(|c| {
                    Complex::new(w(out.gradient[i][c]), w(out.gradient[n + i][c]))
                })
            })
            .collect()
    });
    (potential, gradient)
}

/// The largest errors of `code`'s complex-charge Helmholtz kernel at real k (or, for `None`,
/// the reference `p2p`) on the first [`CHECKED_SETS`] sets of `pool`.
fn accuracy<T: Green>(
    code: Option<Code>,
    k: T,
    pool: &[Set<T>],
    form: Form,
    g: bool,
) -> (f64, f64) {
    let mut worst = (0.0f64, 0.0f64);
    let k = Complex::new(k, T::zero());
    for set in pool.iter().take(CHECKED_SETS) {
        let n = set.targets.len();
        let calls = Calls::new(set, form);
        let mut out = Outputs::new(n);
        let (potential, gradient) = match code {
            Some(code) => {
                evaluate(code, Variant::ComplexQ, k, set, &calls, g, &mut out);
                match code {
                    Code::Green => unpack(g, &out.green_complex[..if g { 4 * n } else { n }]),
                    Code::Ours(_) => from_planes(n, &out, g),
                }
            }
            None => {
                let zero = Complex::new(T::zero(), T::zero());
                let (mut p, mut gr) = (vec![zero; n], vec![[zero; 3]; n]);
                for (sources, _, _, complex) in &calls.calls {
                    let gg = g.then_some(&mut gr[..]);
                    p2p(k, sources, complex, &set.targets, &mut p, gg);
                }
                let w = |v: Complex<T>| Complex::new(v.re.to_f64(), v.im.to_f64());
                (
                    p.iter().map(|&v| w(v)).collect(),
                    g.then(|| gr.iter().map(|x| x.map(w)).collect()),
                )
            }
        };
        let complex: Vec<Complex<T>> = calls.calls.iter().flat_map(|c| c.3.clone()).collect();
        let e = errors(
            k.re.to_f64(),
            set,
            &complex,
            &potential,
            gradient.as_deref(),
        );
        worst = (worst.0.max(e.0), worst.1.max(e.1));
    }
    worst
}

/// The pairs per second of `code` in `variant` on `pool`.
fn time<T: Green>(
    code: Code,
    variant: Variant,
    k: Complex<T>,
    pool: &[Set<T>],
    form: Form,
    g: bool,
) -> f64 {
    let n = pool[0].targets.len();
    let calls: Vec<Calls<T>> = pool.iter().map(|set| Calls::new(set, form)).collect();
    let mut outs: Vec<Outputs<T>> = pool.iter().map(|_| Outputs::new(n)).collect();
    pairs_per_second(pool[0].pairs(), pool.len(), |i| {
        evaluate(code, variant, k, &pool[i], &calls[i], g, &mut outs[i]);
    })
}

/// One code on one cell and output: pairs per second per variant, and the complex-charge
/// kernel's accuracy.
struct Row {
    cell: Cell,
    precision: &'static str,
    gradients: bool,
    code: Code,
    /// Gpairs/s of real q, complex q, damped, Yukawa and Laplace (green-kernels: complex q
    /// and Laplace only, the others NaN).
    rates: [f64; 5],
    errors: (f64, f64),
    reference: (f64, f64),
}

fn measure_cell<T: Green>(cell: Cell, k: Complex<T>) -> Vec<Row> {
    let pool = cell.pool::<T>();
    let form = cell.form();
    let detected = Isa::detect();
    let codes: Vec<Code> = std::iter::once(detected)
        .chain(Isa::available().filter(|&isa| isa != detected))
        .map(Code::Ours)
        .chain([Code::Green])
        .collect();
    let variants = [
        Variant::RealQ,
        Variant::ComplexQ,
        Variant::Damped,
        Variant::Yukawa,
        Variant::Laplace,
    ];
    let mut rows = Vec::new();
    for gradients in [false, true] {
        eprintln!(
            "{} {} {}",
            precision_name::<T>(),
            cell.label(),
            if gradients { "φ, ∇φ" } else { "φ" }
        );
        let reference = accuracy(None, k.re, &pool, form, gradients);
        for &code in &codes {
            let rates = variants.map(|v| match (code, v) {
                (Code::Green, Variant::ComplexQ | Variant::Laplace) | (Code::Ours(_), _) => {
                    time(code, v, k, &pool, form, gradients) / 1e9
                }
                _ => f64::NAN,
            });
            rows.push(Row {
                cell,
                precision: precision_name::<T>(),
                gradients,
                code,
                rates,
                errors: accuracy(Some(code), k.re, &pool, form, gradients),
                reference,
            });
        }
    }
    rows
}

/// The convention check: green-kernels' Helmholtz kernel on one pair, times 4π, against
/// `direct_sum` (potential and ∂/∂x) at real k with a complex charge, in f64. Panics on a
/// mismatch.
fn convention_check(k: f64) -> String {
    let (sources, targets) = ([[0.1, -0.2, 0.3]], [[0.9, 0.4, -0.5]]);
    let charges = [Complex::new(0.7, -0.3)];
    let mut want_p = [Complex::new(0.0, 0.0)];
    let mut want_g = [[Complex::new(0.0, 0.0); 3]];
    direct_sum(
        Complex::new(k, 0.0),
        &sources,
        &charges,
        &targets,
        &mut want_p,
        Some(&mut want_g),
    );
    let mut out = [Complex::new(0.0, 0.0); 4];
    f64::green_helmholtz(k, true, &sources, &charges, &targets, &mut out);
    let (p, g) = unpack(true, &out);
    let g = g.expect("gradients");
    let rel = |a: Complex<f64>, b: Complex<f64>| (a - b).norm() / b.norm();
    let worst = (0..3)
        .map(|c| rel(g[0][c], want_g[0][c]))
        .fold(rel(p[0], want_p[0]), f64::max);
    assert!(
        worst < 1e-10,
        "green-kernels' Helmholtz convention differs: {p:?} {g:?} against {want_p:?} {want_g:?}"
    );
    format!(
        "one pair at k = {k}, complex charge: 4π × green-kernels' φ and ∂/∂x equal \
         `direct_sum` within {worst:.1e} relative (factor 1/(4π), e^{{ikr}}/r, gradient with \
         respect to the target)"
    )
}

fn main() {
    let mut quick = false;
    let (mut re_k, mut im_k) = (1.0f64, 0.1f64);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rest = args.as_slice();
    while let Some((flag, tail)) = rest.split_first() {
        rest = match (flag.as_str(), tail) {
            ("--quick", _) => {
                quick = true;
                tail
            }
            ("--k", [value, tail @ ..]) => {
                re_k = value.parse().expect("--k takes a number");
                tail
            }
            ("--im", [value, tail @ ..]) => {
                im_k = value.parse().expect("--im takes a number");
                tail
            }
            _ => {
                eprintln!("usage: helmholtz [--quick] [--k k] [--im d]; got {args:?}");
                std::process::exit(2);
            }
        };
    }
    let convention = convention_check(re_k);
    let targets: &[usize] = if quick { &[24] } else { &W1_TARGETS };
    let cells: Vec<Cell> = targets
        .iter()
        .flat_map(|&n| [Form::PerPair, Form::Gathered].map(|form| Cell::W1 { targets: n, form }))
        .collect();
    let mut rows = Vec::new();
    for &cell in &cells {
        rows.extend(measure_cell::<f32>(
            cell,
            Complex::new(re_k as f32, im_k as f32),
        ));
    }
    for &cell in &cells {
        rows.extend(measure_cell::<f64>(cell, Complex::new(re_k, im_k)));
    }
    print_header(quick, re_k, im_k, &convention);
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            print_table(&rows, precision, gradients);
        }
    }
    print_summary(&rows);
}

/// The machine, the toolchain, the backends and the method.
fn print_header(quick: bool, re_k: f64, im_k: f64, convention: &str) {
    println!("# Helmholtz P2P: cost per pair against Laplace and Yukawa (Helmholtz CH3.4b)");
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
        "| green-kernels | commit `{GK_COMMIT}`, `Helmholtz3dKernel` and \
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
        "| mode | {}, k = {re_k}, damped k = {re_k} + {im_k}i (leaf-scaled, leaf width 2) |",
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
        "- Gpairs/s counts n_s n_t pairs per evaluation, coincident ones included. \"real q\", \
         \"complex q\": real and complex charges at real k; \"damped\": complex charges at the \
         damped k; Yukawa at κ = Re k. \"cq / L\" and \"cq / Y\": the complex-q kernel's time \
         per pair over the Laplace and the Yukawa kernel's, same code, same cell."
    );
    println!(
        "- Accuracy of the complex-q kernel against `direct_sum` on the first {CHECKED_SETS} \
         sets: the largest error of either part relative to Σ|q|/r (φ) and Σ|q|(1+|k|r)/r² (per \
         component of ∇φ); \"ref\" is the reference `p2p`'s."
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
        "| cell | code | real q | complex q | damped | Yukawa | Laplace | cq / L | cq / Y | max φ | max ∇φ | ref max φ | ref max ∇φ |"
    );
    println!(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    let f = |v: f64| {
        if v.is_nan() {
            "–".to_string()
        } else {
            format!("{v:.3}")
        }
    };
    for row in rows
        .iter()
        .filter(|r| r.precision == precision && r.gradients == gradients)
    {
        let [rq, cq, damped, yukawa, laplace] = row.rates;
        let ratio = |other: f64| {
            if other.is_nan() {
                "–".to_string()
            } else {
                format!("{:.2}", other / cq)
            }
        };
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1e} | {} | {:.1e} | {} |",
            row.cell.label(),
            row.code.name(),
            f(rq),
            f(cq),
            f(damped),
            f(yukawa),
            f(laplace),
            ratio(laplace),
            ratio(yukawa),
            row.errors.0,
            if gradients {
                format!("{:.1e}", row.errors.1)
            } else {
                "–".into()
            },
            row.reference.0,
            if gradients {
                format!("{:.1e}", row.reference.1)
            } else {
                "–".into()
            },
        );
    }
    println!();
}

/// The geometric means over the cells per code, precision and output.
fn print_summary(rows: &[Row]) {
    println!("## Summary: geometric means over the cells");
    println!();
    println!(
        "| precision | output | code | real q / L | cq / L | damped / L | cq / Y | ours cq / green cq |"
    );
    println!("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |");
    let gmean = |v: &[f64]| (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp();
    for precision in ["f32", "f64"] {
        for gradients in [false, true] {
            let sel: Vec<&Row> = rows
                .iter()
                .filter(|r| r.precision == precision && r.gradients == gradients)
                .collect();
            let mut codes: Vec<Code> = Vec::new();
            for r in &sel {
                if !codes.contains(&r.code) {
                    codes.push(r.code);
                }
            }
            for code in codes {
                let mine: Vec<&&Row> = sel.iter().filter(|r| r.code == code).collect();
                let ratio = |a: usize, b: usize| {
                    let v: Vec<f64> = mine.iter().map(|r| r.rates[b] / r.rates[a]).collect();
                    if v.iter().any(|x| x.is_nan()) {
                        "–".to_string()
                    } else {
                        format!("{:.2}", gmean(&v))
                    }
                };
                let versus_green: Vec<f64> = mine
                    .iter()
                    .filter_map(|r| {
                        sel.iter()
                            .find(|g| g.code == Code::Green && g.cell == r.cell)
                            .map(|g| r.rates[1] / g.rates[1])
                    })
                    .collect();
                println!(
                    "| {precision} | {} | {} | {} | {} | {} | {} | {} |",
                    if gradients { "φ, ∇φ" } else { "φ" },
                    code.name(),
                    ratio(0, 4),
                    ratio(1, 4),
                    ratio(2, 4),
                    ratio(1, 3),
                    if code == Code::Green {
                        "–".into()
                    } else {
                        format!("{:.2}", gmean(&versus_green))
                    },
                );
            }
        }
    }
    println!();
    println!(
        "Ratios are times per pair (\"cq / L\" = Laplace's Gpairs/s over complex q's); \
         \"ours cq / green cq\" is ours over green-kernels' Gpairs/s (above 1: ours faster)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The convention check, and one quick cell of each precision: the complex-q kernels
    /// within the term tolerances of the direct sum.
    #[test]
    fn smoke() {
        convention_check(1.0);
        let cell = Cell::W1 {
            targets: 16,
            form: Form::Gathered,
        };
        for g in [false, true] {
            let pool = cell.pool::<f64>();
            let e = accuracy(Some(Code::Ours(Isa::detect())), 1.0, &pool, cell.form(), g);
            assert!(e.0 < 1e-13 && e.1 < 1e-13, "{e:?}");
            let pool = cell.pool::<f32>();
            let e = accuracy(
                Some(Code::Ours(Isa::detect())),
                1.0f32,
                &pool,
                cell.form(),
                g,
            );
            assert!(e.0 < 1e-5 && e.1 < 1e-5, "{e:?}");
        }
    }
}

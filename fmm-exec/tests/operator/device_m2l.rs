//! The device dense M2L (Phase 4 T9, C4.5) as the device operator runs it, the grouped
//! translation of `nd_fmm_kernels::translate` with the 316 dense offset tables, against
//! `nd_fmm_ref::direct` at the absolute frames of CONVENTIONS §3.12, as C2.2 and T8 of
//! Phase 3 check the host operator (`translations.rs`), and against the host operator's
//! own error on the same cell.
//!
//! Geometry, in the dyadic domain: on each level 2, 9 and 16, for every V-list offset d of
//! `m2l_offsets` (all 316 lie inside the domain on these levels), random (source, target)
//! pairs with target − source = d (the canonical frames of §3.12 exactly, asserted). One
//! level call per precision, degree, level and GEMM: one row per target holding its source
//! of offset d (the `Rows` reduction), batches by offset in index order, multipoles and
//! locals in separate buffers as the device operator passes them. Random multipoles whose
//! weighted entries are uniform in [−1, 1], rounded to T first; the reference sees the
//! rounded values.
//!
//! Error measure (`common`): output coefficients per degree in the orthonormal weighting
//! of §3.8 for locals (Nₘ/Sₘ), relative to the terms |A_ik x_k| of the dense f64 table.
//! Bounds (docs/phase4/README.md, "Accuracy measures"; device-path.md §2):
//! - f64 (the CPU runtime), p ∈ {0, 1, 3, 8} (12, 16 and 20 in the ignored sweep): within
//!   1e-14, or within twice the host operator's measured error on the same cell
//!   (`MatrixSet::apply` with the f64 tables against `direct`), whichever is larger;
//! - f32, p ∈ {0, 1, 3, 8}: within 1e-5 against the f64 reference.
//!
//! GEMMs: the hand-written kernel in the backend's default layout and one other (the CPU
//! runtime: its CPU layout and a cube of up to 8 units, correctness only; Metal and CUDA:
//! the cube layout and the CPU layout); on a GPU also `GemmPolicy::Auto` at p = 8 in f32
//! with 128 pairs per offset: on Metal the library GEMM, asserted to run (the library
//! rejects narrow padded shapes such as 16 or 32 columns per offset; the plan reports
//! why), on CUDA the hand-written kernel, the library asserted rejected (its probe fails
//! on the LLVM NVPTX path, device-path.md §18.1, F28).
//!
//! CUDA (Phase 4S T4, ignored, by hand on locust) runs f32 at p ∈ {0, 1, 3, 8} and f64 at
//! every degree, the sweep's included, with the f64 bounds above.
//! Each test prints the device, the worst errors per precision, degree, level and GEMM
//! against the host operator's, and the backends it ran.

use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    Orientation, PlanSettings, Tables, TranslationScratch, grouped,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_fmm_tables::{M2lTables, MatrixSet};

use crate::common::{
    Kind, SplitMix64, absolute_frame, degree_error, dyadic_domain, len, random_coefficients,
    random_pair, terms,
};

/// The precisions of these tests.
trait Real: DeviceFloat + RealScalar {}

impl Real for f32 {}

impl Real for f64 {}

/// The levels of the C2.2 criterion.
const LEVELS: [usize; 3] = [2, 9, 16];

/// The degrees of the default run.
const DEGREES: [usize; 4] = [0, 1, 3, 8];

/// The hand-written layouts of the test on `device` at order n.
fn layouts(device: &Device, n: usize) -> Vec<GemmLayout> {
    let default = GemmLayout::default_for(device.info(), n);
    match device.backend() {
        BackendKind::Cpu => {
            let units = device.info().max_units_per_cube.clamp(1, 8);
            vec![
                default,
                GemmLayout::Cube {
                    rows: (units / 2).max(1),
                    columns: 2.min(units),
                    per_unit: 2,
                },
            ]
        }
        _ => vec![
            default,
            GemmLayout::Cpu {
                block: 8,
                per_unit: 4,
            },
        ],
    }
}

/// The worst errors of one (precision, degree, level, GEMM) cell: the device's and the
/// host operator's, against `direct`, and the GEMM that ran.
#[derive(Clone, Copy, Debug)]
struct Cell {
    device: f64,
    host: f64,
    gemm: Gemm,
}

/// One M2L level call on `level` with `pairs` targets per offset: returns the worst
/// errors, after asserting the bound.
fn run_cell<T: Real>(
    device: &mut Device,
    set: &MatrixSet<f64>,
    (p, level, pairs): (usize, usize, usize),
    layout: GemmLayout,
    policy: GemmPolicy,
    rng: &mut SplitMix64,
) -> Cell {
    let n = len(p);
    let domain = dyadic_domain();
    let offsets = m2l_offsets();
    // Rows: r = d · pairs + j, a target of `level` and its source at offset d.
    let rows = offsets.len() * pairs;
    let mut keys = Vec::with_capacity(rows);
    for (d, &offset) in offsets.iter().enumerate() {
        for _ in 0..pairs {
            let (source, target) = random_pair(level, offset, rng);
            keys.push((d, offset, source, target));
        }
    }
    // One entry per row: (source r, offset d); batch d holds its rows in order.
    let row_offsets: Vec<u32> = (0..=rows as u32).collect();
    let sources: Vec<u32> = (0..rows as u32).collect();
    let groups: Vec<u16> = keys.iter().map(|k| k.0 as u16).collect();
    let batch_offsets: Vec<u32> = (0..=offsets.len()).map(|d| (d * pairs) as u32).collect();
    let arrays = GroupedArrays {
        row_offsets: &row_offsets,
        sources: &sources,
        groups: &groups,
        batch_offsets: &batch_offsets,
        batch_targets: &sources,
        batch_sources: &sources,
    };
    // Inputs rounded to T; the reference sees them as rounded.
    let inputs: Vec<Vec<f64>> = (0..rows)
        .map(|_| {
            random_coefficients(Kind::Multipole, p, rng)
                .into_iter()
                .map(|x| RealScalar::to_f64(T::from_f64(x)))
                .collect()
        })
        .collect();
    let multipoles: Vec<T> = inputs.iter().flatten().map(|&x| T::from_f64(x)).collect();
    let mut locals = vec![T::from_f64(0.0); rows * n];

    let view = GroupedView::upload(device, &arrays, rows).unwrap();
    let settings = PlanSettings {
        n,
        layout,
        policy,
        budget: DEFAULT_SCRATCH_BYTES,
        orientation: Orientation::BoxMajor,
    };
    let library = settings.library_candidate(device.backend(), T::FLOAT);
    let tables = Tables::upload(device, set.cast::<T>().as_slice(), n, library).unwrap();
    let size = settings.size(device.backend(), T::FLOAT, &batch_offsets);
    let mut scratch = TranslationScratch::<T>::new(device, size.columns * n).unwrap();
    let plan = GroupedPlan::new(device, &arrays, rows, &settings, &tables, &mut scratch).unwrap();
    if let Some(reason) = plan.library_rejection() {
        println!("    library not used, the hand-written kernel runs: {reason}");
    }
    if policy == GemmPolicy::Auto && library {
        if device.backend() == BackendKind::Cuda {
            // The library's probe fails on the LLVM NVPTX path (device-path.md §18.1, F28).
            assert!(
                plan.gemm() == Gemm::HandWritten(layout) && plan.library_rejection().is_some(),
                "on CUDA the library must be rejected for [316, {pairs}, {n}]: {}",
                plan.gemm()
            );
        } else {
            assert_eq!(
                plan.gemm(),
                Gemm::Library,
                "the library must take [316, {pairs}, {n}]: {:?}",
                plan.library_rejection()
            );
        }
    }
    let input = device.upload(&multipoles).unwrap();
    let mut output = device.upload(&locals).unwrap();
    grouped(
        device,
        &plan,
        &view,
        Accumulate::Rows,
        &tables,
        Operands::Separate {
            input: input.as_slice(),
            output: output.as_slice_mut(),
        },
        &mut scratch,
    )
    .unwrap();
    device.download(output.as_slice(), &mut locals).unwrap();

    let mut ws = Workspace::new(p);
    let mut cell = Cell {
        device: 0.0,
        host: 0.0,
        gemm: plan.gemm(),
    };
    for (r, &(d, offset, source, target)) in keys.iter().enumerate() {
        let (from, to) = (
            absolute_frame(source, &domain),
            absolute_frame(target, &domain),
        );
        // The canonical frames of §3.12, exactly.
        assert_eq!(from.scaled(to.centre), offset.map(|t| 2.0 * t as f64));
        assert_eq!(to.radius, from.radius);
        let x = &inputs[r];
        let mut want = vec![0.0; n];
        direct::m2l(p, &from, &to, &mut ws, x, &mut want);
        let got: Vec<f64> = locals[r * n..(r + 1) * n]
            .iter()
            .map(|&v| RealScalar::to_f64(v))
            .collect();
        let mut host = vec![0.0; n];
        set.apply(d, x, &mut host);
        let tau = terms(set, d, x);
        let e = degree_error(Kind::Local, p, &got, &want, &tau);
        let h = degree_error(Kind::Local, p, &host, &want, &tau);
        let bound = if T::FLOAT == Precision::F64 {
            1e-14f64.max(2.0 * h)
        } else {
            1e-5
        };
        assert!(
            e <= bound,
            "{} M2L p = {p}, level {level}, offset {d} {offset:?}, {}: {e:.3e} > {bound:.1e} \
             (host operator {h:.3e})",
            T::FLOAT,
            plan.gemm()
        );
        cell.device = cell.device.max(e);
        cell.host = cell.host.max(h);
    }
    cell
}

/// Every degree in `degrees`, level and GEMM in precision T on `device`.
fn check_precision<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = SplitMix64::new(0x7_9800 + T::FLOAT as u64);
    for &p in degrees {
        let set = M2lTables::<f64>::build(p).matrices().clone();
        let n = len(p);
        let mut runs: Vec<(GemmLayout, GemmPolicy, usize)> = layouts(device, n)
            .into_iter()
            .map(|layout| (layout, GemmPolicy::HandWritten, 1))
            .collect();
        if device.backend().is_gpu() && T::FLOAT == Precision::F32 && p == 8 {
            runs.push((
                GemmLayout::default_for(device.info(), n),
                GemmPolicy::Auto,
                128,
            ));
        }
        for (layout, policy, pairs) in runs {
            let cells: Vec<Cell> = LEVELS
                .iter()
                .map(|&level| {
                    run_cell::<T>(device, &set, (p, level, pairs), layout, policy, &mut rng)
                })
                .collect();
            let levels: Vec<String> = LEVELS
                .iter()
                .zip(&cells)
                .map(|(l, c)| format!("{l}: {:.2e} (f64 host {:.2e})", c.device, c.host))
                .collect();
            println!(
                "  {} M2L p = {p:2}, 316 offsets x {pairs} pairs, {}: worst per level {}",
                T::FLOAT,
                cells[0].gemm,
                levels.join(", ")
            );
        }
    }
}

fn run(kind: BackendKind, test: &str, degrees_f64: &[usize], degrees_f32: &[usize]) {
    let mut device = Device::open(kind).unwrap_or_else(|e| panic!("cannot open {kind}: {e}"));
    println!("{test}: on {}", device.info());
    if !degrees_f32.is_empty() {
        check_precision::<f32>(&mut device, degrees_f32);
    }
    if device.supports(Precision::F64) {
        check_precision::<f64>(&mut device, degrees_f64);
    } else {
        println!("  f64: not supported by this device, not run");
    }
    let others: Vec<String> = BackendKind::ALL
        .into_iter()
        .filter(|&b| b != kind)
        .map(|b| {
            let why = match (b, b.is_compiled()) {
                (_, false) => "not compiled",
                (BackendKind::Cpu, true) => "in its own test",
                _ => "in its own ignored test",
            };
            format!("{b} ({why})")
        })
        .collect();
    println!(
        "{test}: backends run: {kind}; not run: {}",
        others.join(", ")
    );
}

#[cfg(feature = "cpu")]
#[test]
fn device_m2l_equals_direct_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_m2l_equals_direct_on_the_cpu_runtime",
        &DEGREES,
        &DEGREES,
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "p up to 20 in f64: run with `--release -- --ignored`"]
fn device_m2l_equals_direct_to_degree_20_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_m2l_equals_direct_to_degree_20_on_the_cpu_runtime",
        &[12, 16, 20],
        &[],
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal: run by hand, outside the sandbox"]
fn device_m2l_equals_direct_on_metal() {
    run(
        BackendKind::Metal,
        "device_m2l_equals_direct_on_metal",
        &[],
        &DEGREES,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA: run by hand on locust"]
fn device_m2l_equals_direct_on_cuda() {
    run(
        BackendKind::Cuda,
        "device_m2l_equals_direct_on_cuda",
        &[0, 1, 3, 8, 12, 16, 20],
        &DEGREES,
    );
}

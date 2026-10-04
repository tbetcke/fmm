//! The device M2M and L2L (Phase 4 T8, C4.4) as the device operator runs them, the grouped
//! translation of `nd_fmm_kernels::translate` with the dense octant tables, against
//! `nd_fmm_ref::direct` at the absolute frames of CONVENTIONS §3.12, as C2.1 and T8 of
//! Phase 3 check the host operators (`translations.rs`), and against the host operator's
//! own error on the same cell.
//!
//! Geometry, in the dyadic domain: on each parent level 1, 2, 9 and 15, for every octant
//! o, two random parents and their child o (the canonical frames of §3.12 exactly,
//! asserted). One level call per kind, level, degree and layout: M2M with one row per
//! parent holding its child of octant o (the `Rows` reduction), L2L with one row per
//! child holding its parent (the `Scatter` accumulation). Random coefficients whose
//! weighted entries are uniform in [−1, 1], rounded to T first; the reference sees the
//! rounded values.
//!
//! Error measure (`common`): output coefficients per degree in the orthonormal weighting
//! of §3.8, relative to the terms |A_ik x_k| of the dense f64 table. Bounds
//! (docs/phase4/README.md, "Accuracy measures"; device-path.md §2):
//! - f64 (the CPU runtime), p ∈ {0, 1, 3, 8} (12, 16 and 20 in the ignored sweep): within
//!   1e-14, or within twice the host operator's measured error on the same cell
//!   (`MatrixSet::apply` with the f64 tables against `direct`), whichever is larger;
//! - f32, p ∈ {0, 1, 3, 8}: within 1e-5 against the f64 reference.
//!
//! Layouts: the backend's default GEMM layout and one other (the CPU runtime: its CPU
//! layout and a cube of up to 8 units, correctness only; Metal: its cube layout and the
//! CPU layout), the hand-written kernel; on Metal also the library GEMM under
//! `GemmPolicy::Auto` at p = 8 with 256 parents per octant (the plan reports whether the
//! library took the shape). Each test prints the device, the worst errors per kind,
//! precision, degree and level against the host operator's, and the backends it ran.

use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    Orientation, PlanSettings, Tables, TranslationScratch, grouped,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::octant_direction;
use nd_fmm_tables::{L2lTables, M2mTables, MatrixSet};
use nd_octree::morton;

use crate::common::{
    Kind, SplitMix64, absolute_frame, degree_error, dyadic_domain, len, random_coefficients, terms,
};

/// The precisions of these tests.
trait Real: DeviceFloat + RealScalar {}

impl Real for f32 {}

impl Real for f64 {}

/// The parent levels of the T8 criterion.
const PARENT_LEVELS: [usize; 4] = [1, 2, 9, 15];

/// The degrees of the default run.
const DEGREES: [usize; 4] = [0, 1, 3, 8];

/// The dense f64 octant tables at p.
struct Octants {
    m2m: MatrixSet<f64>,
    l2l: MatrixSet<f64>,
}

impl Octants {
    fn build(p: usize) -> Self {
        Self {
            m2m: M2mTables::<f64>::build(p).matrices().clone(),
            l2l: L2lTables::<f64>::build(p).matrices().clone(),
        }
    }
}

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

/// The worst errors of one (kind, precision, degree, level) cell: the device's and the
/// host operator's, against `direct`, and the GEMM that ran.
#[derive(Clone, Copy, Debug)]
struct Cell {
    device: f64,
    host: f64,
    gemm: Gemm,
}

/// One level call of M2M (`m2m`) or L2L on `level` (the parents' level) with `pairs`
/// parents per octant: returns the worst errors, after asserting the bound.
#[allow(clippy::too_many_arguments)]
fn run_cell<T: Real>(
    device: &mut Device,
    octants: &Octants,
    (p, level, pairs): (usize, usize, usize),
    m2m: bool,
    layout: GemmLayout,
    policy: GemmPolicy,
    rng: &mut SplitMix64,
) -> Cell {
    let n = len(p);
    let domain = dyadic_domain();
    let set = if m2m { &octants.m2m } else { &octants.l2l };
    let kind = if m2m { Kind::Multipole } else { Kind::Local };
    // Rows: r = o · pairs + j, a parent of `level` and its child o.
    let rows = 8 * pairs;
    let mut keys = Vec::with_capacity(rows);
    for o in 0..8 {
        for _ in 0..pairs {
            let parent = rng.key(level);
            let child = morton::children(parent).unwrap()[o];
            keys.push((o, parent, child));
        }
    }
    // One entry per row: (source r, octant o); batch o holds its rows in order.
    let row_offsets: Vec<u32> = (0..=rows as u32).collect();
    let sources: Vec<u32> = (0..rows as u32).collect();
    let groups: Vec<u8> = keys.iter().map(|k| k.0 as u8).collect();
    let batch_offsets: Vec<u32> = (0..=8).map(|o| (o * pairs) as u32).collect();
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
            random_coefficients(kind, p, rng)
                .into_iter()
                .map(|x| RealScalar::to_f64(T::from_f64(x)))
                .collect()
        })
        .collect();
    let mut store: Vec<T> = inputs.iter().flatten().map(|&x| T::from_f64(x)).collect();
    store.resize(2 * rows * n, T::from_f64(0.0));

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
    let mut buffer = device.upload(&store).unwrap();
    grouped(
        device,
        &plan,
        &view,
        if m2m {
            Accumulate::Rows
        } else {
            Accumulate::Scatter
        },
        &tables,
        Operands::Shared {
            buffer: &mut buffer,
            input: 0..rows * n,
            output: rows * n..2 * rows * n,
        },
        &mut scratch,
    )
    .unwrap();
    device.download(buffer.as_slice(), &mut store).unwrap();

    let mut ws = Workspace::new(p);
    let mut cell = Cell {
        device: 0.0,
        host: 0.0,
        gemm: plan.gemm(),
    };
    for (r, &(o, parent, child)) in keys.iter().enumerate() {
        let (child_frame, parent_frame) = (
            absolute_frame(child, &domain),
            absolute_frame(parent, &domain),
        );
        // The canonical frames of §3.12, exactly.
        let half = octant_direction(o).map(|s| 0.5 * s as f64);
        assert_eq!(parent_frame.scaled(child_frame.centre), half);
        assert_eq!(child_frame.radius / parent_frame.radius, 0.5);
        let x = &inputs[r];
        let mut want = vec![0.0; n];
        if m2m {
            direct::m2m(p, &child_frame, &parent_frame, &mut ws, x, &mut want);
        } else {
            direct::l2l(p, &parent_frame, &child_frame, &mut ws, x, &mut want);
        }
        let got: Vec<f64> = store[(rows + r) * n..(rows + r + 1) * n]
            .iter()
            .map(|&v| RealScalar::to_f64(v))
            .collect();
        let mut host = vec![0.0; n];
        set.apply(o, x, &mut host);
        let tau = terms(set, o, x);
        let e = degree_error(kind, p, &got, &want, &tau);
        let h = degree_error(kind, p, &host, &want, &tau);
        let bound = if T::FLOAT == Precision::F64 {
            1e-14f64.max(2.0 * h)
        } else {
            1e-5
        };
        assert!(
            e <= bound,
            "{} {} p = {p}, parent level {level}, octant {o}, {}: {e:.3e} > {bound:.1e} \
             (host operator {h:.3e})",
            T::FLOAT,
            if m2m { "M2M" } else { "L2L" },
            plan.gemm()
        );
        cell.device = cell.device.max(e);
        cell.host = cell.host.max(h);
    }
    cell
}

/// Every kind, degree in `degrees`, level and layout in precision T on `device`.
fn check_precision<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = SplitMix64::new(0x7_8800 + T::FLOAT as u64);
    for &p in degrees {
        let octants = Octants::build(p);
        let n = len(p);
        let mut runs: Vec<(GemmLayout, GemmPolicy, usize)> = layouts(device, n)
            .into_iter()
            .map(|layout| (layout, GemmPolicy::HandWritten, 2))
            .collect();
        if device.backend().is_gpu() && T::FLOAT == Precision::F32 && p == 8 {
            runs.push((
                GemmLayout::default_for(device.info(), n),
                GemmPolicy::Auto,
                256,
            ));
        }
        for (layout, policy, pairs) in runs {
            for m2m in [true, false] {
                let cells: Vec<Cell> = PARENT_LEVELS
                    .iter()
                    .map(|&level| {
                        run_cell::<T>(
                            device,
                            &octants,
                            (p, level, pairs),
                            m2m,
                            layout,
                            policy,
                            &mut rng,
                        )
                    })
                    .collect();
                let levels: Vec<String> = PARENT_LEVELS
                    .iter()
                    .zip(&cells)
                    .map(|(l, c)| format!("{l}: {:.2e} (f64 host {:.2e})", c.device, c.host))
                    .collect();
                println!(
                    "  {} {} p = {p:2}, {} parents per octant, {}: worst per parent level {}",
                    T::FLOAT,
                    if m2m { "M2M" } else { "L2L" },
                    pairs,
                    cells[0].gemm,
                    levels.join(", ")
                );
            }
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
                (BackendKind::Cuda, true) => "type-checked, not run",
                _ => "in its own test",
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
fn device_translations_equal_direct_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_translations_equal_direct_on_the_cpu_runtime",
        &DEGREES,
        &DEGREES,
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "p up to 20 in f64: run with `--release -- --ignored`"]
fn device_translations_equal_direct_to_degree_20_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_translations_equal_direct_to_degree_20_on_the_cpu_runtime",
        &[12, 16, 20],
        &[],
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal: run by hand, outside the sandbox"]
fn device_translations_equal_direct_on_metal() {
    run(
        BackendKind::Metal,
        "device_translations_equal_direct_on_metal",
        &[],
        &DEGREES,
    );
}

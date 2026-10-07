//! The device rotation M2L (Phase 4 T10, C4.6) as the device operator runs it,
//! `nd_fmm_kernels::rotation::m2l` with the M2L family of `RotationTables` (uploaded through
//! `nd_fmm_exec::device::RotationHostArrays`), against `nd_fmm_ref::direct` at the absolute
//! frames of CONVENTIONS §3.12, as C2.3 and T8 of Phase 3 check the host operator
//! (`translations.rs`), against the host's `RotationTables::m2l` in f64, and against the
//! device dense M2L of T9 on the same batches.
//!
//! Geometry, in the dyadic domain: on each level 2, 9 and 16, for every V-list offset d of
//! `m2l_offsets` (all 316 lie inside the domain on these levels), a random (source, target)
//! pair with target − source = d (the canonical frames of §3.12 exactly, asserted). One
//! level call per precision, degree, level and layout: one row per target holding its
//! source of offset d, multipoles and locals in separate buffers as the device operator
//! passes them, the locals from zero. Random multipoles whose weighted entries are uniform
//! in [−1, 1], rounded to T first; the references see the rounded values. The 316 offsets
//! include the four on the z axis (the coaxial step alone) and every polar angle, the
//! smallest and the largest included.
//!
//! Error measure (`common`): output coefficients per degree in the orthonormal weighting
//! of §3.8 for locals (Nₘ/Sₘ), relative to the terms |A_ik x_k| of the dense f64 table.
//! Bounds (docs/phase4/README.md, "Accuracy measures"; T10 brief):
//! - f64 (the CPU runtime), p ∈ {0, 1, 3, 8} (12, 16 and 20 in the ignored sweep): within
//!   1e-13 of `direct` and of the host's `RotationTables::m2l` in f64;
//! - f32, p ∈ {0, 1, 3, 8}: within 1e-5 of `direct` and of the host's f64 rotation;
//! - against the device dense M2L (`translate::grouped` with the 316 tables of
//!   `M2lTables` and the hand-written GEMM, the device default) on the same batches in the
//!   same precision: within 1e-13 (f64) and 1e-5 (f32).
//!
//! Layouts: the backend's default and one other (the CPU runtime: its CPU layout and a cube
//! of at most (p + 1)² units, as many as the device allows, correctness only; Metal and
//! CUDA: the cube layout and the CPU layout). CUDA (Phase 4S T4, ignored, by hand on
//! locust) runs f32 at p ∈ {0, 1, 3, 8} and f64 at every degree, the sweep's included,
//! with the f64 bounds above. Each test prints the device, the worst errors per
//! precision, degree, level and layout against the host operator's own error, and the
//! backends it ran.

use nd_fmm_exec::device::RotationHostArrays;
use nd_fmm_kernels::rotation::{RotationLayout, RotationPlan, RotationTables, m2l};
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, GemmLayout, GemmPolicy, GroupedPlan, Operands, Orientation,
    PlanSettings, Tables, TranslationScratch, grouped,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_fmm_tables::rotation::Operator;
use nd_fmm_tables::{M2lTables, MatrixSet, RotationScratch};

use crate::common::{
    Kind, SplitMix64, absolute_frame, degree_error, dyadic_domain, len, random_coefficients,
    random_pair, terms,
};

/// The precisions of these tests.
trait Real: DeviceFloat + RealScalar + nd_fmm_exec::device::DeviceScalar {}

impl Real for f32 {}

impl Real for f64 {}

/// The levels of the C2.3 criterion.
const LEVELS: [usize; 3] = [2, 9, 16];

/// The degrees of the default run.
const DEGREES: [usize; 4] = [0, 1, 3, 8];

/// The layouts of the test on `device` at degree p.
fn layouts(device: &Device, p: usize) -> Vec<RotationLayout> {
    let default = RotationLayout::default_for(device.info(), p);
    match device.backend() {
        BackendKind::Cpu => vec![
            default,
            RotationLayout::Cube {
                units: (len(p) as u32).min(device.info().max_units_per_cube.max(1)),
            },
        ],
        _ => vec![default, RotationLayout::Cpu],
    }
}

/// The tables of one degree: the dense f64 tables (terms, and the device dense M2L), and
/// the rotation tables in f64 (the host reference) and in T (the device).
struct Degree<T: Real> {
    p: usize,
    dense: MatrixSet<f64>,
    host: nd_fmm_tables::RotationTables<f64>,
    arrays: RotationHostArrays<T>,
}

impl<T: Real> Degree<T> {
    fn new(p: usize) -> Self {
        let host = nd_fmm_tables::RotationTables::<f64>::build(p);
        let arrays = RotationHostArrays::new(&host.tables(Operator::M2l).cast::<T>());
        Self {
            p,
            dense: M2lTables::<f64>::build(p).matrices().clone(),
            host,
            arrays,
        }
    }
}

/// The worst errors of one (precision, degree, level, layout) cell, per degree in the
/// terms: the device against `direct`, against the host rotation and against the device
/// dense M2L; the host rotation against `direct`.
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    direct: f64,
    host: f64,
    dense: f64,
    host_direct: f64,
}

/// The rows of a level call: one per (offset, pair), with its keys.
struct Batch<T> {
    keys: Vec<(
        usize,
        nd_octree::morton::MortonKey,
        nd_octree::morton::MortonKey,
    )>,
    inputs: Vec<Vec<f64>>,
    multipoles: Vec<T>,
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u16>,
    batch_offsets: Vec<u32>,
}

impl<T: Real> Batch<T> {
    fn new(p: usize, level: usize, rng: &mut SplitMix64) -> Self {
        let offsets = m2l_offsets();
        let mut keys = Vec::with_capacity(offsets.len());
        for (d, &offset) in offsets.iter().enumerate() {
            let (source, target) = random_pair(level, offset, rng);
            keys.push((d, source, target));
        }
        let rows = keys.len();
        let inputs: Vec<Vec<f64>> = (0..rows)
            .map(|_| {
                random_coefficients(Kind::Multipole, p, rng)
                    .into_iter()
                    .map(|x| RealScalar::to_f64(T::from_f64(x)))
                    .collect()
            })
            .collect();
        Self {
            multipoles: inputs.iter().flatten().map(|&x| T::from_f64(x)).collect(),
            inputs,
            row_offsets: (0..=rows as u32).collect(),
            sources: (0..rows as u32).collect(),
            groups: keys.iter().map(|k| k.0 as u16).collect(),
            batch_offsets: (0..=offsets.len() as u32).collect(),
            keys,
        }
    }

    fn arrays(&self) -> GroupedArrays<'_, u16> {
        GroupedArrays {
            row_offsets: &self.row_offsets,
            sources: &self.sources,
            groups: &self.groups,
            batch_offsets: &self.batch_offsets,
            batch_targets: &self.sources,
            batch_sources: &self.sources,
        }
    }

    fn rows(&self) -> usize {
        self.keys.len()
    }

    /// The device rotation M2L in `layout` from zero locals.
    fn rotation(&self, device: &mut Device, degree: &Degree<T>, layout: RotationLayout) -> Vec<T> {
        let n = len(degree.p);
        let tables = RotationTables::upload(device, &degree.arrays.arrays()).unwrap();
        let view = GroupedView::upload(device, &self.arrays(), self.rows()).unwrap();
        let plan = RotationPlan::new(device, &self.arrays(), self.rows()).unwrap();
        let input = device.upload(&self.multipoles).unwrap();
        let mut output = device.alloc::<T>(self.rows() * n).unwrap();
        let before = device.counters().launches;
        m2l(
            device,
            layout,
            &plan,
            &view,
            &tables,
            input.as_slice(),
            output.as_slice_mut(),
        )
        .unwrap();
        assert_eq!(device.counters().launches - before, 1, "one launch");
        let mut locals = vec![T::from_f64(0.0); self.rows() * n];
        device.download(output.as_slice(), &mut locals).unwrap();
        locals
    }

    /// The device dense M2L (T9) with the hand-written GEMM in the backend's default
    /// layout, from zero locals.
    fn dense(&self, device: &mut Device, degree: &Degree<T>) -> Vec<T> {
        let n = len(degree.p);
        let settings = PlanSettings {
            n,
            layout: GemmLayout::default_for(device.info(), n),
            policy: GemmPolicy::HandWritten,
            budget: DEFAULT_SCRATCH_BYTES,
            orientation: Orientation::BoxMajor,
        };
        let tables = Tables::upload(device, degree.dense.cast::<T>().as_slice(), n, false).unwrap();
        let view = GroupedView::upload(device, &self.arrays(), self.rows()).unwrap();
        let size = settings.size(device.backend(), T::FLOAT, &self.batch_offsets);
        let mut scratch = TranslationScratch::<T>::new(device, size.columns * n).unwrap();
        let plan = GroupedPlan::new(
            device,
            &self.arrays(),
            self.rows(),
            &settings,
            &tables,
            &mut scratch,
        )
        .unwrap();
        let input = device.upload(&self.multipoles).unwrap();
        let mut output = device.alloc::<T>(self.rows() * n).unwrap();
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
        let mut locals = vec![T::from_f64(0.0); self.rows() * n];
        device.download(output.as_slice(), &mut locals).unwrap();
        locals
    }
}

/// One level of degree p in every layout: asserts the bounds, returns the worst errors per
/// layout.
fn run_level<T: Real>(
    device: &mut Device,
    degree: &Degree<T>,
    level: usize,
    rng: &mut SplitMix64,
) -> Vec<(RotationLayout, Cell)> {
    let p = degree.p;
    let n = len(p);
    let domain = dyadic_domain();
    let batch = Batch::<T>::new(p, level, rng);
    let dense = batch.dense(device, degree);
    let bound = if T::FLOAT == Precision::F64 {
        1e-13
    } else {
        1e-5
    };
    // The references per row: direct, the host rotation in f64, and the terms.
    let mut ws = Workspace::new(p);
    let mut scratch = RotationScratch::new(p);
    let offsets = m2l_offsets();
    let references: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = batch
        .keys
        .iter()
        .zip(&batch.inputs)
        .map(|(&(d, source, target), x)| {
            let (from, to) = (
                absolute_frame(source, &domain),
                absolute_frame(target, &domain),
            );
            // The canonical frames of §3.12, exactly.
            assert_eq!(from.scaled(to.centre), offsets[d].map(|t| 2.0 * t as f64));
            assert_eq!(to.radius, from.radius);
            let mut want = vec![0.0; n];
            direct::m2l(p, &from, &to, &mut ws, x, &mut want);
            let mut host = vec![0.0; n];
            degree.host.m2l(d, x, &mut host, &mut scratch);
            (want, host, terms(&degree.dense, d, x))
        })
        .collect();
    let widen = |v: &[T]| -> Vec<f64> { v.iter().map(|&x| RealScalar::to_f64(x)).collect() };
    let mut cells = Vec::new();
    for layout in layouts(device, p) {
        let locals = batch.rotation(device, degree, layout);
        let mut cell = Cell::default();
        for (r, (want, host, tau)) in references.iter().enumerate() {
            let (d, ..) = batch.keys[r];
            let got = widen(&locals[r * n..(r + 1) * n]);
            let other = widen(&dense[r * n..(r + 1) * n]);
            let errors = [
                degree_error(Kind::Local, p, &got, want, tau),
                degree_error(Kind::Local, p, &got, host, tau),
                degree_error(Kind::Local, p, &got, &other, tau),
            ];
            for (what, e) in ["direct", "the host rotation", "the device dense M2L"]
                .iter()
                .zip(errors)
            {
                assert!(
                    e <= bound,
                    "{} rotation M2L p = {p}, level {level}, offset {d} {:?}, {layout}: \
                     {e:.3e} from {what} > {bound:.0e}",
                    T::FLOAT,
                    offsets[d]
                );
            }
            cell.direct = cell.direct.max(errors[0]);
            cell.host = cell.host.max(errors[1]);
            cell.dense = cell.dense.max(errors[2]);
            cell.host_direct = cell
                .host_direct
                .max(degree_error(Kind::Local, p, host, want, tau));
        }
        cells.push((layout, cell));
    }
    cells
}

/// Every degree in `degrees`, level and layout in precision T on `device`.
fn check_precision<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = SplitMix64::new(0x7_a800 + T::FLOAT as u64);
    for &p in degrees {
        let degree = Degree::<T>::new(p);
        let per_level: Vec<Vec<(RotationLayout, Cell)>> = LEVELS
            .iter()
            .map(|&level| run_level(device, &degree, level, &mut rng))
            .collect();
        for (i, layout) in layouts(device, p).into_iter().enumerate() {
            let levels: Vec<String> = LEVELS
                .iter()
                .zip(&per_level)
                .map(|(l, cells)| {
                    let c = cells[i].1;
                    format!(
                        "{l}: {:.2e} (host rotation {:.2e}; from it {:.2e}, from dense {:.2e})",
                        c.direct, c.host_direct, c.host, c.dense
                    )
                })
                .collect();
            println!(
                "  {} rotation M2L p = {p:2}, 316 offsets, {layout}: worst against direct per \
                 level {}",
                T::FLOAT,
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
fn device_rotation_equals_direct_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_rotation_equals_direct_on_the_cpu_runtime",
        &DEGREES,
        &DEGREES,
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "p up to 20 in f64: run with `--release -- --ignored`"]
fn device_rotation_equals_direct_to_degree_20_on_the_cpu_runtime() {
    run(
        BackendKind::Cpu,
        "device_rotation_equals_direct_to_degree_20_on_the_cpu_runtime",
        &[12, 16, 20],
        &[],
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal: run by hand, outside the sandbox"]
fn device_rotation_equals_direct_on_metal() {
    run(
        BackendKind::Metal,
        "device_rotation_equals_direct_on_metal",
        &[],
        &DEGREES,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA: run by hand on locust"]
fn device_rotation_equals_direct_on_cuda() {
    run(
        BackendKind::Cuda,
        "device_rotation_equals_direct_on_cuda",
        &[0, 1, 3, 8, 12, 16, 20],
        &DEGREES,
    );
}

//! The grouped translations (T8, C4.4): the hand-written GEMM, the grouped level call and
//! its chunks, on every layout ([`GemmLayout`]) the backend runs, in f32 and, where the
//! device supports it, f64; on Metal also the library GEMM.
//!
//! - **GEMM**: y = A x for one table of order n = (p + 1)², p ∈ {0, 1, 3, 8, 12, 20},
//!   k ∈ {0, 1, 7, 64} columns (k = 1000 in the ignored test on the CPU runtime, and on
//!   Metal): bit for bit a host loop `acc = x_k.mul_add(a_ik, acc)` from zero, k
//!   ascending (the kernel's documented order; spikes/device-arith/REPORT.md, rule 6), and
//!   within n u_T of `MatrixSet::apply` relative to the terms Σₖ |A_ik x_k|.
//! - **Level calls** (M2M-shaped rows with up to eight octants, some rows empty; L2L-shaped
//!   rows of at most one entry): bit for bit the host emulation (each target: load, then
//!   for each entry of its row in row order add the fma-loop product, store), within the
//!   GEMM tolerance of applying every (target, source, octant) of the rows in row order
//!   with `MatrixSet::apply`; rows without entries untouched, bit for bit; the same bits
//!   with every scratch budget (one chunk or many); repeated calls bit-identical.
//! - **Grouped against per-octant**: a level call equals one gather, GEMM and scatter-add
//!   per octant in octant order (structure (A) of device-path.md §6.4), bit for bit.
//! - **Library** (Metal, f32, p = 8): the plan of a level call under `GemmPolicy::Auto`
//!   reports its GEMM; with the library, the call is within the GEMM tolerance of the host
//!   rows and bit-identical when repeated.
//!
//! u_T = 2⁻²⁴ (f32), 2⁻⁵³ (f64). Values are uniform in [−1, 1] (normal, so a flushing
//! backend adds them exactly as the host).

use nd_fmm_kernels::movement::{gather_columns, scatter_add_columns};
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    Orientation, PlanSettings, Tables, TileSchedule, TranslationScratch, gemm, grouped,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::MatrixSet;

use crate::common::{Rng, TestFloat, tests_on};

/// The float types of the tests.
trait Real: TestFloat + RealScalar {}

impl Real for f32 {}

impl Real for f64 {}

fn w<T: Real>(x: T) -> f64 {
    <T as RealScalar>::to_f64(x)
}

fn t<T: Real>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// The unit roundoff u_T.
fn unit<T: Real>() -> f64 {
    if T::FLOAT == Precision::F64 {
        2f64.powi(-53)
    } else {
        2f64.powi(-24)
    }
}

/// Uniform in [−1, 1).
fn uniform<T: Real>(rng: &mut Rng, len: usize) -> Vec<T> {
    (0..len)
        .map(|_| t::<T>(2.0 * ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64) - 1.0))
        .collect()
}

/// The precisions `device` runs, with a function per precision.
macro_rules! each_precision {
    ($device:expr, $body:ident) => {{
        $body::<f32>($device);
        if $device.supports(Precision::F64) {
            $body::<f64>($device);
        } else {
            println!("  f64: not supported by this device, not run");
        }
    }};
}

/// The hand-written layouts a backend runs in these tests: on the CPU runtime its default
/// and a small cube layout (at most 8 units: one per core on the CI runner, correctness
/// only); on a GPU its default, a 1-D cube and the CPU layout.
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
                    per_unit: 3,
                },
            ]
        }
        _ => vec![
            default,
            GemmLayout::Cube {
                rows: 64,
                columns: 1,
                per_unit: 2,
            },
            GemmLayout::Cpu {
                block: 4,
                per_unit: 3,
            },
        ],
    }
}

/// The product of the column-major table `a` (order n) with `x`, as the kernel forms it:
/// one accumulator per output from zero, `acc = x_k.mul_add(a_ik, acc)` for k ascending.
fn host_product<T: Real>(a: &[T], x: &[T], n: usize) -> Vec<T> {
    (0..n)
        .map(|i| {
            (0..n).fold(<T as RealScalar>::from_f64(0.0), |acc, k| {
                x[k].mul_add(a[k * n + i], acc)
            })
        })
        .collect()
}

/// Σₖ |A_ik x_k| in f64.
fn terms<T: Real>(a: &[T], x: &[T], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|k| (w(a[k * n + i]) * w(x[k])).abs()).sum())
        .collect()
}

/// The values of `buffer`.
fn download<T: Real>(device: &mut Device, buffer: &DeviceBuffer<T>) -> Vec<T> {
    let mut out = vec![t::<T>(0.0); buffer.len()];
    device.download(buffer.as_slice(), &mut out).unwrap();
    out
}

/// Asserts `got` equals `want` bit for bit.
fn assert_same<T: Real>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g.bits() != w.bits()) {
        let count = got
            .iter()
            .zip(want)
            .filter(|(g, w)| g.bits() != w.bits())
            .count();
        panic!(
            "{what}: {count} of {} values differ, the first at {i}: {:?} against {:?}",
            got.len(),
            got[i],
            want[i]
        );
    }
}

// --- The GEMM ---------------------------------------------------------------------------

/// One GEMM of k columns with the table of order n in `layout`: bit for bit the host
/// product, within n u_T of `MatrixSet::apply`. Returns the largest error in units of
/// u_T times the terms.
fn check_gemm<T: Real>(device: &mut Device, layout: GemmLayout, p: usize, k: usize) -> f64 {
    let n = (p + 1) * (p + 1);
    let mut rng = Rng::new(0x7_8000 + 31 * p as u64 + k as u64);
    let mut set = MatrixSet::<T>::zeros(n, 1);
    set.matrix_mut(0)
        .copy_from_slice(&uniform::<T>(&mut rng, n * n));
    let x_host = uniform::<T>(&mut rng, n * k);
    let tables = device.upload(set.as_slice()).unwrap();
    let x = device.upload(&x_host).unwrap();
    // The output starts from a pattern the kernel must overwrite, with a column of slack.
    let mut y = device.upload(&vec![t::<T>(7.0); n * (k + 1)]).unwrap();
    let schedule = TileSchedule::new(device, layout, &[0, k as u32]).unwrap();
    let before = device.counters().launches;
    gemm(
        device,
        layout,
        n,
        tables.as_slice(),
        &schedule,
        x.as_slice(),
        y.as_slice_mut(),
    )
    .unwrap();
    assert_eq!(
        device.counters().launches - before,
        u64::from(k > 0),
        "one launch, none without columns"
    );
    let got = download(device, &y);
    let what = format!("{} p = {p}, k = {k}, {layout}", T::FLOAT);
    let mut worst = 0.0f64;
    for c in 0..k {
        let column = &x_host[c * n..(c + 1) * n];
        assert_same(
            &format!("{what}, column {c}: the host fma loop"),
            &got[c * n..(c + 1) * n],
            &host_product(set.matrix(0), column, n),
        );
        let mut apply = vec![t::<T>(0.0); n];
        set.apply(0, column, &mut apply);
        let tau = terms(set.matrix(0), column, n);
        for i in 0..n {
            let e = (w(got[c * n + i]) - w(apply[i])).abs();
            let bound = n as f64 * unit::<T>() * tau[i];
            assert!(
                e <= bound,
                "{what}, column {c}, row {i}: |device − apply| = {e:e} > n u_T τ = {bound:e}"
            );
            if tau[i] > 0.0 {
                worst = worst.max(e / (unit::<T>() * tau[i]));
            }
        }
    }
    assert!(
        got[k * n..].iter().all(|&v| w(v) == 7.0),
        "{what}: the slack column was written"
    );
    worst
}

fn gemm_sweep<T: Real>(device: &mut Device, columns: &[usize]) {
    for p in [0usize, 1, 3, 8, 12, 20] {
        let n = (p + 1) * (p + 1);
        for layout in layouts(device, n) {
            let worst = columns
                .iter()
                .map(|&k| check_gemm::<T>(device, layout, p, k))
                .fold(0.0, f64::max);
            println!(
                "  {} p = {p:2}, {layout}: k ∈ {columns:?} bit for bit the host fma loop; \
                 |device − MatrixSet::apply| ≤ {worst:.2} u_T τ (bound n = {n})",
                T::FLOAT
            );
        }
    }
}

fn gemm_small<T: Real>(device: &mut Device) {
    gemm_sweep::<T>(device, &[0, 1, 7, 64]);
}

fn gemm_large<T: Real>(device: &mut Device) {
    gemm_sweep::<T>(device, &[1000]);
}

fn gemm_equals_the_host_loop(device: &mut Device) {
    each_precision!(device, gemm_small);
}

fn gemm_of_1000_columns(device: &mut Device) {
    each_precision!(device, gemm_large);
}

// --- Level calls ------------------------------------------------------------------------

/// A grouped view of `nrows` targets over 8 groups (octants): M2M-shaped rows (`rows_shape`
/// false: a random subset of the octants per row, some rows empty) or L2L-shaped rows (at
/// most one entry per row), sources among `nsources`. The six arrays of `GroupedArrays`.
struct View {
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u8>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
    nsources: usize,
}

impl View {
    fn random(rng: &mut Rng, nrows: usize, nsources: usize, one_per_row: bool) -> Self {
        let mut rows: Vec<Vec<(u8, u32)>> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let mut row = Vec::new();
            // Every fifth row is empty (a leaf or a ghost).
            if r % 5 != 4 {
                if one_per_row {
                    row.push((rng.below(8) as u8, rng.below(nsources) as u32));
                } else {
                    for o in 0..8u8 {
                        if rng.below(3) > 0 {
                            row.push((o, rng.below(nsources) as u32));
                        }
                    }
                }
            }
            rows.push(row);
        }
        let mut view = Self {
            row_offsets: vec![0],
            sources: Vec::new(),
            groups: Vec::new(),
            batch_offsets: vec![0],
            batch_targets: Vec::new(),
            batch_sources: Vec::new(),
            nsources,
        };
        for row in &rows {
            for &(g, s) in row {
                view.groups.push(g);
                view.sources.push(s);
            }
            view.row_offsets.push(view.sources.len() as u32);
        }
        for g in 0..8u8 {
            for (target, row) in rows.iter().enumerate() {
                if let Some(&(_, s)) = row.iter().find(|e| e.0 == g) {
                    view.batch_targets.push(target as u32);
                    view.batch_sources.push(s);
                }
            }
            view.batch_offsets.push(view.batch_targets.len() as u32);
        }
        view
    }

    fn arrays(&self) -> GroupedArrays<'_, u8> {
        GroupedArrays {
            row_offsets: &self.row_offsets,
            sources: &self.sources,
            groups: &self.groups,
            batch_offsets: &self.batch_offsets,
            batch_targets: &self.batch_targets,
            batch_sources: &self.batch_sources,
        }
    }

    fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    fn upload(&self, device: &mut Device) -> GroupedView {
        GroupedView::upload(device, &self.arrays(), self.nsources).unwrap()
    }
}

/// The host emulation of a level call: for each target in row order, load, add the
/// fma-loop product of each entry in row order, store (rows without entries untouched).
fn host_rows<T: Real>(view: &View, set: &MatrixSet<T>, input: &[T], output: &[T]) -> Vec<T> {
    let n = set.n();
    let mut out = output.to_vec();
    for r in 0..view.nrows() {
        let entries = view.row_offsets[r] as usize..view.row_offsets[r + 1] as usize;
        for e in entries {
            let (g, s) = (view.groups[e] as usize, view.sources[e] as usize);
            let y = host_product(set.matrix(g), &input[s * n..(s + 1) * n], n);
            for i in 0..n {
                out[r * n + i] += y[i];
            }
        }
    }
    out
}

/// The rows applied with `MatrixSet::apply` in row order, and the terms of each output
/// (|initial value| plus Σ over its entries of Σₖ |A_ik x_k|).
fn apply_rows<T: Real>(
    view: &View,
    set: &MatrixSet<T>,
    input: &[T],
    output: &[T],
) -> (Vec<T>, Vec<f64>) {
    let n = set.n();
    let mut out = output.to_vec();
    let mut tau: Vec<f64> = output.iter().map(|&v| w(v).abs()).collect();
    for r in 0..view.nrows() {
        for e in view.row_offsets[r] as usize..view.row_offsets[r + 1] as usize {
            let (g, s) = (view.groups[e] as usize, view.sources[e] as usize);
            let x = &input[s * n..(s + 1) * n];
            set.apply(g, x, &mut out[r * n..(r + 1) * n]);
            for (i, v) in terms(set.matrix(g), x, n).into_iter().enumerate() {
                tau[r * n + i] += v;
            }
        }
    }
    (out, tau)
}

/// The data of a level call: tables, view, and one buffer holding the input columns, then
/// the output columns (the M2M and L2L shape, `Operands::Shared`).
struct Call<T: Real> {
    set: MatrixSet<T>,
    view: View,
    input: Vec<T>,
    output: Vec<T>,
}

impl<T: Real> Call<T> {
    fn new(seed: u64, n: usize, nrows: usize, one_per_row: bool) -> Self {
        let mut rng = Rng::new(seed);
        let mut set = MatrixSet::<T>::zeros(n, 8);
        for g in 0..8 {
            set.matrix_mut(g)
                .copy_from_slice(&uniform::<T>(&mut rng, n * n));
        }
        let nsources = 2 * nrows + 3;
        let view = View::random(&mut rng, nrows, nsources, one_per_row);
        let input = uniform::<T>(&mut rng, nsources * n);
        let output = uniform::<T>(&mut rng, nrows * n);
        Self {
            set,
            view,
            input,
            output,
        }
    }

    fn accumulate(&self) -> Accumulate {
        if self.view.row_offsets.windows(2).all(|w| w[1] - w[0] <= 1) {
            Accumulate::Scatter
        } else {
            Accumulate::Rows
        }
    }

    /// Runs the level call with `settings` and returns the output and the plan's GEMM
    /// and chunk count. `repeat` runs it a second time from the same output and checks
    /// the bits.
    fn run(
        &self,
        device: &mut Device,
        settings: &PlanSettings,
        repeat: bool,
    ) -> (Vec<T>, Gemm, usize) {
        let n = self.set.n();
        let library = settings.library_candidate(device.backend(), T::FLOAT);
        let tables = Tables::upload(device, self.set.as_slice(), n, library).unwrap();
        let view = self.view.upload(device);
        let size = settings.size(device.backend(), T::FLOAT, &self.view.batch_offsets);
        let mut scratch = TranslationScratch::<T>::new(device, size.columns * n).unwrap();
        let plan = GroupedPlan::new(
            device,
            &self.view.arrays(),
            self.view.nsources,
            settings,
            &tables,
            &mut scratch,
        )
        .unwrap();
        assert!(plan.columns() <= size.columns);
        if let Some(reason) = plan.library_rejection() {
            println!("    library not used, the hand-written kernel runs: {reason}");
        }
        let initial: Vec<T> = self.input.iter().chain(&self.output).copied().collect();
        let mut store = device.upload(&initial).unwrap();
        let (input, output) = (0..self.input.len(), self.input.len()..initial.len());
        let mut results = Vec::new();
        for _ in 0..1 + usize::from(repeat) {
            device.write(store.as_slice_mut(), &initial).unwrap();
            let before = device.counters().launches;
            grouped(
                device,
                &plan,
                &view,
                self.accumulate(),
                &tables,
                Operands::Shared {
                    buffer: &mut store,
                    input: input.clone(),
                    output: output.clone(),
                },
                &mut scratch,
            )
            .unwrap();
            assert_eq!(
                device.counters().launches - before,
                3 * plan.nchunks() as u64,
                "three launches per chunk"
            );
            let all = download(device, &store);
            assert_same("the input", &all[input.clone()], &self.input);
            results.push(all[output.clone()].to_vec());
        }
        if repeat {
            assert_same("a repeated level call", &results[1], &results[0]);
        }
        (results.swap_remove(0), plan.gemm(), plan.nchunks())
    }
}

/// The level-call checks of the module documentation for one shape, on every layout.
fn check_level_call<T: Real>(device: &mut Device, p: usize, nrows: usize, one_per_row: bool) {
    let n = (p + 1) * (p + 1);
    let call = Call::<T>::new(
        0x7_8100 + 7 * p as u64 + nrows as u64,
        n,
        nrows,
        one_per_row,
    );
    let shape = if one_per_row { "L2L" } else { "M2M" };
    let want = host_rows(&call.view, &call.set, &call.input, &call.output);
    let (apply, tau) = apply_rows(&call.view, &call.set, &call.input, &call.output);
    for layout in layouts(device, n) {
        let what = format!("{} {shape} p = {p}, {nrows} rows, {layout}", T::FLOAT);
        let mut chunks = Vec::new();
        // One chunk; chunks of 5 columns (cutting batches); chunks of 1 column; each in both
        // orientations (T12: the same products, so the same bits).
        for columns in [usize::MAX, 5, 1] {
            let budget = if columns == usize::MAX {
                DEFAULT_SCRATCH_BYTES
            } else {
                (2 * n * columns * size_of::<T>()) as u64
            };
            for orientation in [Orientation::BoxMajor, Orientation::CoefficientMajor] {
                let settings = PlanSettings {
                    n,
                    layout,
                    policy: GemmPolicy::HandWritten,
                    budget,
                    orientation,
                };
                let (got, gemm, nchunks) = call.run(device, &settings, columns == usize::MAX);
                assert_eq!(gemm, Gemm::HandWritten(layout));
                if orientation == Orientation::BoxMajor {
                    chunks.push(nchunks);
                }
                assert_same(
                    &format!("{what}, {orientation}, {nchunks} chunks: the host rows"),
                    &got,
                    &want,
                );
            }
        }
        let pairs = call.view.sources.len();
        assert_eq!(chunks[0], usize::from(pairs > 0));
        assert_eq!(chunks[2], pairs);
        // Within the GEMM tolerance of `MatrixSet::apply` in row order: per output n u_T
        // for the product and one rounding per entry, relative to its terms.
        let mut worst = 0.0f64;
        for (i, (&a, &tau)) in apply.iter().zip(&tau).enumerate() {
            let e = (w(want[i]) - w(a)).abs();
            let bound = (n + 9) as f64 * unit::<T>() * tau;
            assert!(e <= bound, "{what}, value {i}: {e:e} > {bound:e}");
            if tau > 0.0 {
                worst = worst.max(e / (unit::<T>() * tau));
            }
        }
        println!(
            "  {what}: bit for bit the host rows with 1, {} and {} chunks, box-major and \
             coefficient-major, repeated calls bit-identical, empty rows untouched; \
             ≤ {worst:.2} u_T τ from apply",
            chunks[1], chunks[2]
        );
    }
}

/// Structure (A) of device-path.md §6.4: per octant in octant order one gather, GEMM and
/// scatter-add into the output; equal to the grouped call bit for bit.
fn check_per_group<T: Real>(device: &mut Device, p: usize) {
    let n = (p + 1) * (p + 1);
    let call = Call::<T>::new(0x7_8200 + p as u64, n, 40, false);
    for layout in layouts(device, n) {
        let settings = PlanSettings {
            n,
            layout,
            policy: GemmPolicy::HandWritten,
            budget: DEFAULT_SCRATCH_BYTES,
            orientation: Orientation::BoxMajor,
        };
        let (grouped_result, _, _) = call.run(device, &settings, false);
        let tables = device.upload(call.set.as_slice()).unwrap();
        let view = call.view.upload(device);
        let input = device.upload(&call.input).unwrap();
        let mut output = device.upload(&call.output).unwrap();
        let widest = call
            .view
            .batch_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap();
        let mut x = device.alloc::<T>(widest.max(1) * n).unwrap();
        let mut y = device.alloc::<T>(widest.max(1) * n).unwrap();
        for g in 0..8 {
            let batch =
                call.view.batch_offsets[g] as usize..call.view.batch_offsets[g + 1] as usize;
            let k = batch.len();
            if k == 0 {
                continue;
            }
            gather_columns(
                device,
                n,
                input.as_slice(),
                view.batch_sources().slice(batch.clone()),
                x.slice_mut(..k * n),
            )
            .unwrap();
            let schedule = TileSchedule::new(device, layout, &[0, k as u32]).unwrap();
            gemm(
                device,
                layout,
                n,
                tables.slice(g * n * n..(g + 1) * n * n),
                &schedule,
                x.slice(..k * n),
                y.slice_mut(..k * n),
            )
            .unwrap();
            scatter_add_columns(
                device,
                n,
                y.slice(..k * n),
                view.batch_targets().slice(batch),
                output.as_slice_mut(),
            )
            .unwrap();
        }
        let per_group = download(device, &output);
        assert_same(
            &format!("{} p = {p}, {layout}: grouped against per octant", T::FLOAT),
            &grouped_result,
            &per_group,
        );
        println!(
            "  {} p = {p}, {layout}: the grouped call equals per-octant gather, GEMM and \
             scatter-add bit for bit",
            T::FLOAT
        );
    }
}

fn level_calls<T: Real>(device: &mut Device) {
    for p in [0usize, 3, 8] {
        check_level_call::<T>(device, p, 40, false);
        check_level_call::<T>(device, p, 40, true);
    }
    // A level without pairs launches nothing and leaves the output as it is.
    let mut call = Call::<T>::new(0x7_8150, 16, 6, true);
    call.view = View::random(&mut Rng::new(1), 6, call.view.nsources, true);
    call.view.row_offsets = vec![0; 7];
    call.view.sources.clear();
    call.view.groups.clear();
    call.view.batch_offsets = vec![0; 9];
    call.view.batch_targets.clear();
    call.view.batch_sources.clear();
    let settings = PlanSettings {
        n: 16,
        layout: GemmLayout::default_for(device.info(), 16),
        policy: GemmPolicy::Auto,
        budget: DEFAULT_SCRATCH_BYTES,
        orientation: Orientation::BoxMajor,
    };
    let (got, _, chunks) = call.run(device, &settings, true);
    assert_eq!(chunks, 0);
    assert_same("a level without pairs", &got, &call.output);
}

fn per_group<T: Real>(device: &mut Device) {
    for p in [0usize, 3, 8] {
        check_per_group::<T>(device, p);
    }
}

fn level_calls_equal_the_host_rows(device: &mut Device) {
    each_precision!(device, level_calls);
}

fn grouped_equals_per_octant(device: &mut Device) {
    each_precision!(device, per_group);
}

/// The library GEMM under `GemmPolicy::Auto` at p = 8 in f32 (Metal), in both orientations
/// (T12): the plan's GEMM is reported; with the library, a level call is within the GEMM
/// tolerance of the host rows and bit-identical when repeated. Elsewhere the plan must run
/// the hand-written kernel.
fn library_level_call(device: &mut Device) {
    let p = 8;
    let n = (p + 1) * (p + 1);
    let shapes = [(40, false), (40, true), (2000, false)];
    for ((nrows, one_per_row), orientation) in shapes.into_iter().flat_map(|shape| {
        [Orientation::BoxMajor, Orientation::CoefficientMajor].map(|o| (shape, o))
    }) {
        let call = Call::<f32>::new(0x7_8300 + nrows as u64, n, nrows, one_per_row);
        let settings = PlanSettings {
            n,
            layout: GemmLayout::default_for(device.info(), n),
            policy: GemmPolicy::Auto,
            budget: DEFAULT_SCRATCH_BYTES,
            orientation,
        };
        let (got, gemm, _) = call.run(device, &settings, true);
        let (apply, tau) = apply_rows(&call.view, &call.set, &call.input, &call.output);
        let mut worst = 0.0f64;
        for (i, (&a, &tau)) in apply.iter().zip(&tau).enumerate() {
            let e = (w(got[i]) - w(a)).abs();
            let bound = (n + 9) as f64 * unit::<f32>() * tau;
            assert!(e <= bound, "library p = 8, value {i}: {e:e} > {bound:e}");
            if tau > 0.0 {
                worst = worst.max(e / (unit::<f32>() * tau));
            }
        }
        if !device.backend().is_gpu() {
            assert!(matches!(gemm, Gemm::HandWritten(_)), "{gemm}");
        }
        println!(
            "  f32 p = 8, {nrows} rows ({}), {orientation}: GEMM {gemm}; ≤ {worst:.2} u_T τ \
             from apply, repeated calls bit-identical",
            if one_per_row { "L2L" } else { "M2M" }
        );
    }
}

tests_on!(cpu: gemm_equals_the_host_loop, level_calls_equal_the_host_rows,
    grouped_equals_per_octant, library_level_call);

/// The ignored CPU-runtime GEMM of 1000 columns (fmm-kernels/CLAUDE.md, test budget).
#[cfg(feature = "cpu")]
#[test]
#[ignore = "large k on the CPU runtime: run with -- --ignored"]
fn gemm_of_1000_columns_on_the_cpu_runtime() {
    crate::common::run(
        BackendKind::Cpu,
        "translate::gemm_of_1000_columns_on_the_cpu_runtime",
        gemm_of_1000_columns,
    );
}

mod gpu {
    use super::*;

    fn gemm_on_metal(device: &mut Device) {
        gemm_equals_the_host_loop(device);
        gemm_of_1000_columns(device);
    }

    tests_on!(metal: gemm_on_metal, level_calls_equal_the_host_rows,
        grouped_equals_per_octant, library_level_call);
}

/// Dense M2L (T9, C4.5): level calls over V-list-shaped views of the 316 offsets.
mod m2l;

/// Rotation M2L (T10, C4.6): level calls over V-list-shaped views with the rotation tables.
mod rotation;

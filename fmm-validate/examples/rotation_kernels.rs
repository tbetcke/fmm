//! The device rotation M2L of `nd-fmm-kernels` (Phase 4 T10, C4.6) against the device dense
//! M2L of T9, per pair across p, on the V batches of the C3.2 cube; its achieved flop rate
//! against the break-even efficiencies of the GEMM spike; and the whole M2L stage on the
//! device against the host rotation at 1 and 12 threads. Prints Markdown on stdout and
//! progress on stderr. Feature `gpu` (with a backend).
//!
//! The problem is the C3.2 cube (N = 10⁵, a uniform level-4 tree, 4,096 leaves), in f32.
//! The host `Fmm` is built once, for its plan (the V views do not depend on p); the
//! multipoles are seeded random coefficients (the timing does not depend on the values).
//! For each p ∈ {2, 4, 6, 8, 12, 16} (`--degrees` to change), per level with a V pair:
//!
//! - **rotation**: the level call as the device operator runs it under `Rotation`
//!   (`nd_fmm_kernels::rotation::m2l` with the M2L family of `RotationTables`, the
//!   backend's default layout: on Metal a cube of (p + 1)² units rounded up to the plane
//!   size); one launch;
//! - **dense**: the T9 level call as the device operator runs it under `Dense` with the
//!   default `DeviceGemm::Auto` (`translate::grouped` with the 316 tables of `M2lTables` and
//!   the hand-written GEMM in the backend's default layout; three launches per chunk), and
//!   its GEMM alone; where the library takes the level's shapes (f32, p ≥ 8, a GPU), also
//!   the level call with the library GEMM (`DeviceGemm::Library`), the faster of the two
//!   being "dense, best";
//! - first, the rotation's locals are compared with the dense ones (relative L2 over the
//!   level), so that a timed row is a correct one.
//!
//! Flops per pair, as the spike counts them (spikes/cubecl-gemm/SPIKE_REPORT.md, "Dense
//! against rotation M2L in f64"): (20/3)(p + 1)³ for rotation (2 per multiply–add of
//! (10/3)(p + 1)³; a model: the four offsets on the z axis need less, and the exact count
//! of `ShiftTables::apply` is (p + 1)(10p² + 32p + 9)/3 multiply–adds), 2 (p + 1)⁴ for
//! dense. % of peak against 14.3 TFLOP/s (the M3 Max GPU in f32, derived in the spike
//! report, not measured). The rotation efficiency is compared with the spike's
//! break-even efficiencies: the measured f32 case on the M3 Max (13.9%, 9.8%, 7.1% and
//! 4.8% at p = 4, 8, 12 and 16: rotation would match the spike's best dense GEMM at
//! B = 10⁵ there), and the f64 model for data-centre cards (A100 and H100: 13.2%, 5.4% and
//! 3.8% at p = 8, 12 and 16, central). The f64 comparison is a model statement: no f64 GPU
//! run is possible here (Metal has no f64; CUDA is type-checked only).
//!
//! **The M2L stage at p = 8** (every V level, f32): the device rotation against the host
//! rotation (`LaplaceOperator::m2l_pair` with the f32 rotation tables, the body of its level
//! call) on one thread and on `--threads` scoped threads (default 12, the performance
//! cores), each a contiguous share of the level's boxes, with every BLAS thread variable
//! set to 1 by the launcher; the device result is checked against the host's (relative L2).
//!
//! **Layouts at p = 8** (the deepest V level): the rotation cube with 96 (the default),
//! 128 and 192 units, and a cube of 32 units (three slots per unit). The CPU layout, one
//! cube of at most 16 units on a GPU, is left out: one call took 9.5 s on Metal.
//!
//! Timing: before each table the device runs half a second of other work (its clocks ramp
//! down while the host builds); launches are queued between syncs, after a warm-up launch
//! and a sync (compilation excluded); each figure is the median of 15 batches of at least
//! 20 ms. Nothing is asserted.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example rotation_kernels -- \
//!     --device metal [--table-cache DIR]
//! ```
//!
//! `--device cpu` runs the same on the CubeCL CPU runtime (correctness backend; its
//! timings are not a Phase 4 target; use `--quick` there). `--quick` runs p = 4 only and
//! skips the stage and the layouts. `--table-cache DIR` loads and stores the tables there
//! (`nd_fmm_tables::TableCache`). Metal needs a process with GPU access (outside the macOS
//! sandbox). It initialises MPI and runs on one rank.

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::device::RotationHostArrays;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables as HostTables};
use nd_fmm_kernels::rotation::{RotationLayout, RotationPlan, RotationTables, m2l};
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    PlanSettings, Stage, Tables, TranslationScratch, grouped, grouped_stage,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_tables::rotation::Operator;
use nd_fmm_tables::{M2lTables, TableCache};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Problem};

/// The f32 peak of the M3 Max GPU (spikes/cubecl-gemm/SPIKE_REPORT.md, derived).
const PEAK_GFLOPS: f64 = 14_300.0;

/// The spike's break-even rotation efficiencies (fraction of peak): the measured f32 case
/// on the M3 Max (best dense GEMM, B = 10⁵), and the f64 model for the A100 and H100
/// (central; the H100's within 0.1 point of the A100's), per p.
const BREAK_EVEN_F32_M3: [(usize, f64); 4] = [(4, 0.139), (8, 0.098), (12, 0.071), (16, 0.048)];
const BREAK_EVEN_F64_MODEL: [(usize, f64); 3] = [(8, 0.132), (12, 0.054), (16, 0.038)];

/// The command line.
struct Arguments {
    device: String,
    threads: usize,
    quick: bool,
    degrees: Vec<usize>,
    table_cache: Option<String>,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: rotation_kernels --device cpu|metal|cuda [--threads n] [--quick] \
             [--degrees p,p,...] [--table-cache DIR]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        threads: 12,
        quick: false,
        degrees: vec![2, 4, 6, 8, 12, 16],
        table_cache: None,
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--device", [value, tail @ ..]) => {
                parsed.device = value.clone();
                tail
            }
            ("--threads", [value, tail @ ..]) => {
                parsed.threads = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            ("--degrees", [value, tail @ ..]) => {
                parsed.degrees = value
                    .split(',')
                    .map(|p| p.parse().unwrap_or_else(|_| usage()))
                    .collect();
                tail
            }
            ("--table-cache", [value, tail @ ..]) => {
                parsed.table_cache = Some(value.clone());
                tail
            }
            ("--quick", _) => {
                parsed.quick = true;
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty() || parsed.threads == 0 || parsed.degrees.is_empty() {
        usage();
    }
    if parsed.quick {
        parsed.degrees = vec![4];
    }
    parsed
}

/// The V view of one level of the plan, copied, with the box keys of the level.
struct Level {
    level: usize,
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u16>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
    keys: Vec<u64>,
}

impl Level {
    fn arrays(&self) -> GroupedArrays<'_, u16> {
        GroupedArrays {
            row_offsets: &self.row_offsets,
            sources: &self.sources,
            groups: &self.groups,
            batch_offsets: &self.batch_offsets,
            batch_targets: &self.batch_targets,
            batch_sources: &self.batch_sources,
        }
    }

    fn boxes(&self) -> usize {
        self.keys.len()
    }

    fn pairs(&self) -> usize {
        self.sources.len()
    }
}

/// The V views of every level with a pair.
fn levels(fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>) -> Vec<Level> {
    let plan = fmm.plan();
    let index = plan.index();
    let mut out = Vec::new();
    for (level, lists) in plan.levels().iter().enumerate() {
        let v = lists.v();
        if v.is_empty() {
            continue;
        }
        out.push(Level {
            level,
            row_offsets: v.row_offsets().to_vec(),
            sources: v.sources().to_vec(),
            groups: v.groups().to_vec(),
            batch_offsets: v.batch_offsets().to_vec(),
            batch_targets: v.batch_targets().to_vec(),
            batch_sources: v.batch_sources().to_vec(),
            keys: index.keys(level).to_vec(),
        });
    }
    out
}

/// The tables of one degree in f32: the M2L family of the rotation tables and the 316 dense
/// M2L matrices (with a cache, both load).
struct DegreeTables {
    p: usize,
    rotation: nd_fmm_tables::RotationTables<f32>,
    dense: Vec<f32>,
}

impl DegreeTables {
    fn new(p: usize, cache: Option<&TableCache>) -> Self {
        eprintln!("p = {p}: building the tables");
        let (rotation, dense): (nd_fmm_tables::RotationTables<f32>, M2lTables<f32>) = match cache {
            Some(cache) => (cache.load_or_build(p).0, cache.load_or_build(p).0),
            None => (nd_fmm_tables::RotationTables::build(p), M2lTables::build(p)),
        };
        Self {
            p,
            rotation,
            dense: dense.matrices().as_slice().to_vec(),
        }
    }

    fn n(&self) -> usize {
        (self.p + 1) * (self.p + 1)
    }
}

/// Median seconds of `run`, queued between syncs.
fn time(device: &mut Device, mut run: impl FnMut(&mut Device)) -> f64 {
    run(device);
    device.sync().unwrap();
    median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            run(device);
        }
        device.sync().unwrap();
        start.elapsed()
    })
}

/// Keeps the device busy for half a second, so that the first timed launch does not meet a
/// GPU clocked down while the host built the tables.
fn warm_up(device: &mut Device) {
    let mut buffer = device.alloc::<f32>(1 << 22).unwrap();
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < 0.5 {
        for _ in 0..16 {
            nd_fmm_kernels::movement::zero(device, buffer.as_slice_mut()).unwrap();
        }
        device.sync().unwrap();
    }
}

/// The relative L2 difference of `a` from `b`.
fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        r += f64::from(y).powi(2);
    }
    if r > 0.0 { (d / r).sqrt() } else { d.sqrt() }
}

/// Rotation flops per pair, as the spike counts them: (20/3)(p + 1)³.
fn rotation_flops(p: usize) -> f64 {
    20.0 / 3.0 * ((p + 1) as f64).powi(3)
}

/// Dense flops per pair: 2 (p + 1)⁴.
fn dense_flops(p: usize) -> f64 {
    2.0 * ((p + 1) as f64).powi(4)
}

/// The data of one level on the device: the view, the rotation plan, the multipoles and
/// the locals.
struct DeviceLevel {
    view: GroupedView,
    plan: RotationPlan,
    multipoles: DeviceBuffer<f32>,
    multipoles_host: Vec<f32>,
    locals: DeviceBuffer<f32>,
}

impl DeviceLevel {
    fn new(device: &mut Device, level: &Level, n: usize, rng: &mut SplitMix64) -> Self {
        let boxes = level.boxes();
        let multipoles_host: Vec<f32> = (0..boxes * n)
            .map(|i| rng.range(-1.0, 1.0) as f32 / (1.0 + (i % n) as f32))
            .collect();
        Self {
            view: GroupedView::upload(device, &level.arrays(), boxes).unwrap(),
            plan: RotationPlan::new(device, &level.arrays(), boxes).unwrap(),
            multipoles: device.upload(&multipoles_host).unwrap(),
            multipoles_host,
            locals: device.alloc::<f32>(boxes * n).unwrap(),
        }
    }

    /// One rotation level call in `layout`.
    fn rotation(
        &mut self,
        device: &mut Device,
        layout: RotationLayout,
        tables: &RotationTables<f32>,
    ) {
        m2l(
            device,
            layout,
            &self.plan,
            &self.view,
            tables,
            self.multipoles.as_slice(),
            self.locals.as_slice_mut(),
        )
        .unwrap();
    }

    /// The locals after one call of `run` from zero.
    fn locals_after(
        &mut self,
        device: &mut Device,
        run: impl FnOnce(&mut Device, &mut Self),
    ) -> Vec<f32> {
        let zeros = vec![0.0f32; self.locals.len()];
        device.write(self.locals.as_slice_mut(), &zeros).unwrap();
        run(device, self);
        let mut out = zeros;
        device.download(self.locals.as_slice(), &mut out).unwrap();
        out
    }
}

/// One timed level at one degree.
struct Row {
    level: usize,
    boxes: usize,
    pairs: usize,
    rotation: f64,
    dense: f64,
    dense_gemm: f64,
    dense_chunks: usize,
    library: Option<f64>,
    difference: f64,
}

/// Times every V level at the degree of `tables`.
fn measure(device: &mut Device, views: &[Level], tables: &DegreeTables) -> Vec<Row> {
    let (p, n) = (tables.p, tables.n());
    let host_arrays = RotationHostArrays::new(tables.rotation.tables(Operator::M2l));
    let rotation_tables = RotationTables::upload(device, &host_arrays.arrays()).unwrap();
    let layout = RotationLayout::default_for(device.info(), p);
    let hand_settings = PlanSettings {
        n,
        layout: GemmLayout::default_for(device.info(), n),
        policy: GemmPolicy::HandWritten,
        budget: DEFAULT_SCRATCH_BYTES,
    };
    let library_settings = PlanSettings {
        policy: GemmPolicy::Auto,
        ..hand_settings
    };
    let library_copy = library_settings.library_candidate(device.backend(), Precision::F32);
    let dense_tables = Tables::upload(device, &tables.dense, n, library_copy).unwrap();
    let mut rng = SplitMix64::new(0x7e_a001 + p as u64);
    warm_up(device);
    let mut rows = Vec::new();
    for level in views {
        eprintln!("p = {p}: M2L on level {}", level.level);
        let mut data = DeviceLevel::new(device, level, n, &mut rng);
        let plan_of = |device: &mut Device, settings: &PlanSettings| {
            let size = settings.size(device.backend(), Precision::F32, &level.batch_offsets);
            let mut scratch = TranslationScratch::<f32>::new(device, size.columns * n).unwrap();
            let plan = GroupedPlan::new(
                device,
                &level.arrays(),
                level.boxes(),
                settings,
                &dense_tables,
                &mut scratch,
            )
            .unwrap();
            (plan, scratch)
        };
        let dense_call = |device: &mut Device,
                          data: &mut DeviceLevel,
                          plan: &GroupedPlan,
                          scratch: &mut TranslationScratch<f32>,
                          stage: Option<Stage>| {
            let operands = Operands::Separate {
                input: data.multipoles.as_slice(),
                output: data.locals.as_slice_mut(),
            };
            match stage {
                None => grouped(
                    device,
                    plan,
                    &data.view,
                    Accumulate::Rows,
                    &dense_tables,
                    operands,
                    scratch,
                ),
                Some(stage) => grouped_stage(
                    device,
                    plan,
                    &data.view,
                    Accumulate::Rows,
                    &dense_tables,
                    operands,
                    scratch,
                    stage,
                ),
            }
            .unwrap();
        };
        let (hand_plan, mut hand_scratch) = plan_of(device, &hand_settings);
        // The check: the rotation against the dense level call, from zero.
        let rotated =
            data.locals_after(device, |d, data| data.rotation(d, layout, &rotation_tables));
        let dense = data.locals_after(device, |d, data| {
            dense_call(d, data, &hand_plan, &mut hand_scratch, None);
        });
        let difference = relative_l2(&rotated, &dense);
        let rotation = time(device, |d| data.rotation(d, layout, &rotation_tables));
        let dense_time = time(device, |d| {
            dense_call(d, &mut data, &hand_plan, &mut hand_scratch, None);
        });
        let dense_gemm = time(device, |d| {
            dense_call(
                d,
                &mut data,
                &hand_plan,
                &mut hand_scratch,
                Some(Stage::Gemm),
            );
        });
        let library = if library_copy {
            let (plan, mut scratch) = plan_of(device, &library_settings);
            (plan.gemm() == Gemm::Library).then(|| {
                time(device, |d| {
                    dense_call(d, &mut data, &plan, &mut scratch, None)
                })
            })
        } else {
            None
        };
        rows.push(Row {
            level: level.level,
            boxes: level.boxes(),
            pairs: level.pairs(),
            rotation,
            dense: dense_time,
            dense_gemm,
            dense_chunks: hand_plan.nchunks(),
            library,
            difference,
        });
    }
    rows
}

fn print_header(device: &Device, arguments: &Arguments) {
    println!("# The device rotation M2L against dense (Phase 4 T10, C4.6)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {} |", device.info());
    println!(
        "| problem | the C3.2 cube: N = 10⁵ uniform in a cube, a uniform level-4 tree; its V \
         levels 2 to 4 |"
    );
    println!(
        "| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a \
         warm-up launch and a sync first (compilation excluded) |"
    );
    println!(
        "| flops | rotation (20/3)(p + 1)³ per pair (the spike's count), dense 2 (p + 1)⁴ per \
         pair; peak {PEAK_GFLOPS} GFLOP/s (M3 Max GPU f32, derived in the spike report, not \
         measured) |"
    );
    println!(
        "| host | `LaplaceOperator::m2l_pair` with the f32 rotation tables, 1 and {} scoped \
         threads |",
        arguments.threads
    );
    println!(
        "| precision | f32 (f64 not timed: Metal has no f64, CUDA is type-checked only; the \
         f64 comparison is a model) |"
    );
    println!();
}

fn print_degree(p: usize, rows: &[Row], device: &Device) {
    let rotation_rate = |t: f64, pairs: usize| rotation_flops(p) * pairs as f64 / t / 1e9;
    let dense_rate = |t: f64, pairs: usize| dense_flops(p) * pairs as f64 / t / 1e9;
    println!(
        "## p = {p} (n = {}), {}, rotation layout {}",
        (p + 1) * (p + 1),
        device.backend(),
        RotationLayout::default_for(device.info(), p)
    );
    println!();
    println!(
        "| level | boxes | pairs | rotation µs | ns/pair | GFLOP/s (model) | % peak | dense \
         level call µs (chunks) | ns/pair | of which GEMM µs | GFLOP/s | library call µs | \
         rotation / dense best | relative L2 rotation – dense |"
    );
    println!(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: \
         | ---: |"
    );
    for r in rows {
        let best = r.library.map_or(r.dense, |l| l.min(r.dense));
        let g = rotation_rate(r.rotation, r.pairs);
        println!(
            "| {} | {} | {} | {:.1} | {:.2} | {:.0} | {:.2} | {:.1} ({}) | {:.2} | {:.1} | {:.0} | \
             {} | {:.2} | {:.1e} |",
            r.level,
            r.boxes,
            r.pairs,
            r.rotation * 1e6,
            r.rotation / r.pairs as f64 * 1e9,
            g,
            100.0 * g / PEAK_GFLOPS,
            r.dense * 1e6,
            3 * r.dense_chunks,
            r.dense / r.pairs as f64 * 1e9,
            r.dense_gemm * 1e6,
            dense_rate(r.dense_gemm, r.pairs),
            r.library
                .map_or("–".to_owned(), |l| format!("{:.1}", l * 1e6)),
            r.rotation / best,
            r.difference,
        );
    }
    println!();
}

/// The summary across p: the whole M2L stage (every V level) per pair.
fn print_summary(totals: &[(usize, Vec<Row>)]) {
    println!("## Rotation against dense across p (every V level of the cube, f32)");
    println!();
    println!(
        "| p | pairs | rotation µs | ns/pair | % peak (model flops) | dense (hand-written) µs | \
         ns/pair | dense, best µs | rotation / dense best | break-even, M3 f32 measured | \
         break-even, A100/H100 f64 model |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |");
    for (p, rows) in totals {
        let pairs: usize = rows.iter().map(|r| r.pairs).sum();
        let rotation: f64 = rows.iter().map(|r| r.rotation).sum();
        let dense: f64 = rows.iter().map(|r| r.dense).sum();
        let best: f64 = rows
            .iter()
            .map(|r| r.library.map_or(r.dense, |l| l.min(r.dense)))
            .sum();
        let efficiency = rotation_flops(*p) * pairs as f64 / rotation / 1e9 / PEAK_GFLOPS;
        let lookup = |table: &[(usize, f64)]| {
            table
                .iter()
                .find(|(q, _)| q == p)
                .map_or("–".to_owned(), |(_, e)| {
                    format!(
                        "{:.1}% ({})",
                        100.0 * e,
                        if efficiency >= *e {
                            "rotation at or above"
                        } else {
                            "rotation below"
                        }
                    )
                })
        };
        println!(
            "| {p} | {pairs} | {:.1} | {:.2} | {:.2}% | {:.1} | {:.2} | {:.1} | {:.2} | {} | {} |",
            rotation * 1e6,
            rotation / pairs as f64 * 1e9,
            100.0 * efficiency,
            dense * 1e6,
            dense / pairs as f64 * 1e9,
            best * 1e6,
            rotation / best,
            lookup(&BREAK_EVEN_F32_M3),
            lookup(&BREAK_EVEN_F64_MODEL),
        );
    }
    println!();
}

/// The host rotation M2L of `level` from zero locals on `threads` scoped threads, each a
/// contiguous share of the boxes: its locals and its median time per call.
fn host_level(
    level: &Level,
    p: usize,
    multipoles: &[f32],
    host: &LaplaceOperator<f32>,
    threads: usize,
) -> (Vec<f32>, f64) {
    let n = (p + 1) * (p + 1);
    let boxes = level.boxes();
    let mut operators: Vec<LaplaceOperator<f32>> = (0..threads).map(|_| host.clone()).collect();
    let mut out = vec![0.0f32; boxes * n];
    let run = |out: &mut [f32], operators: &mut [LaplaceOperator<f32>]| {
        out.fill(0.0);
        let mut rest = out;
        let mut pieces = Vec::with_capacity(threads);
        for k in 0..threads {
            let (start, end) = (k * boxes / threads, (k + 1) * boxes / threads);
            let (piece, tail) = std::mem::take(&mut rest).split_at_mut((end - start) * n);
            pieces.push((start..end, piece));
            rest = tail;
        }
        std::thread::scope(|scope| {
            for ((share, piece), op) in pieces.into_iter().zip(operators.iter_mut()) {
                scope.spawn(move || {
                    let first = share.start;
                    for t in share {
                        let row = level.row_offsets[t] as usize..level.row_offsets[t + 1] as usize;
                        let local = &mut piece[(t - first) * n..(t - first + 1) * n];
                        for e in row {
                            let s = level.sources[e] as usize;
                            op.m2l_pair(
                                level.keys[s],
                                level.keys[t],
                                &multipoles[s * n..(s + 1) * n],
                                local,
                            );
                        }
                    }
                });
            }
        });
    };
    run(&mut out, &mut operators);
    let result = out.clone();
    let seconds = median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            run(&mut out, &mut operators);
        }
        start.elapsed()
    });
    (result, seconds)
}

/// The M2L stage at p = 8: the device rotation against the host rotation (module
/// documentation).
fn stage(device: &mut Device, views: &[Level], tables: &DegreeTables, threads: usize) {
    let (p, n) = (tables.p, tables.n());
    let host_tables = HostTables::<f32>::build(p, M2lStrategy::Rotation);
    let host = LaplaceOperator::new(host_tables, false, 64);
    let host_arrays = RotationHostArrays::new(tables.rotation.tables(Operator::M2l));
    let rotation_tables = RotationTables::upload(device, &host_arrays.arrays()).unwrap();
    let layout = RotationLayout::default_for(device.info(), p);
    let mut rng = SplitMix64::new(0x7e_a100);
    warm_up(device);
    println!("## The M2L stage at p = {p}: device rotation against host rotation (f32)");
    println!();
    println!(
        "| level | pairs | device µs | host 1 thread µs | host {threads} threads µs | x host 1 | \
         x host {threads} | relative L2 device – host |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    let (mut sum_device, mut sum_one, mut sum_many, mut pairs) = (0.0, 0.0, 0.0, 0);
    for level in views {
        eprintln!("stage: level {}", level.level);
        let mut data = DeviceLevel::new(device, level, n, &mut rng);
        let got = data.locals_after(device, |d, data| data.rotation(d, layout, &rotation_tables));
        let device_time = time(device, |d| data.rotation(d, layout, &rotation_tables));
        let (want, one) = host_level(level, p, &data.multipoles_host, &host, 1);
        let (_, many) = host_level(level, p, &data.multipoles_host, &host, threads);
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1e} |",
            level.level,
            level.pairs(),
            device_time * 1e6,
            one * 1e6,
            many * 1e6,
            one / device_time,
            many / device_time,
            relative_l2(&got, &want)
        );
        sum_device += device_time;
        sum_one += one;
        sum_many += many;
        pairs += level.pairs();
    }
    println!(
        "| all | {pairs} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | |",
        sum_device * 1e6,
        sum_one * 1e6,
        sum_many * 1e6,
        sum_one / sum_device,
        sum_many / sum_device
    );
    println!();
}

/// The rotation layouts at p = 8 on the deepest V level (module documentation).
fn layouts(device: &mut Device, level: &Level, tables: &DegreeTables) {
    let (p, n) = (tables.p, tables.n());
    let host_arrays = RotationHostArrays::new(tables.rotation.tables(Operator::M2l));
    let rotation_tables = RotationTables::upload(device, &host_arrays.arrays()).unwrap();
    let mut rng = SplitMix64::new(0x7e_a200);
    let mut data = DeviceLevel::new(device, level, n, &mut rng);
    let default = RotationLayout::default_for(device.info(), p);
    let reference = data.locals_after(device, |d, data| {
        data.rotation(d, default, &rotation_tables)
    });
    warm_up(device);
    println!(
        "## Rotation layouts at p = {p}, level {} ({} pairs)",
        level.level,
        level.pairs()
    );
    println!();
    println!("| layout | µs | ns/pair | % peak (model) | bit for bit the default |");
    println!("| --- | ---: | ---: | ---: | --- |");
    let mut candidates = vec![default];
    if device.backend().is_gpu() {
        candidates.extend([
            RotationLayout::Cube { units: 128 },
            RotationLayout::Cube { units: 192 },
            RotationLayout::Cube { units: 32 },
        ]);
    }
    for layout in candidates {
        if let Err(error) = layout.check(device.info(), p, Precision::F32) {
            println!("| {layout} | refused: {error} | | | |");
            continue;
        }
        let got = data.locals_after(device, |d, data| data.rotation(d, layout, &rotation_tables));
        let seconds = time(device, |d| data.rotation(d, layout, &rotation_tables));
        let g = rotation_flops(p) * level.pairs() as f64 / seconds / 1e9;
        println!(
            "| {layout} | {:.1} | {:.2} | {:.2} | {} |",
            seconds * 1e6,
            seconds / level.pairs() as f64 * 1e9,
            100.0 * g / PEAK_GFLOPS,
            if got
                .iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            {
                "yes"
            } else {
                "no"
            }
        );
    }
    println!();
}

fn main() {
    let arguments = arguments();
    let Some(kind) = BackendKind::from_name(&arguments.device) else {
        eprintln!("--device {}: expected cpu, metal or cuda", arguments.device);
        std::process::exit(2);
    };
    let mut device = Device::open(kind).unwrap_or_else(|error| {
        eprintln!(
            "--device {}: {error} (Metal needs a process with GPU access)",
            arguments.device
        );
        std::process::exit(1);
    });
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let comm = universe.world();
    let cache = arguments.table_cache.as_deref().map(TableCache::new);
    print_header(&device, &arguments);
    let config = Config::C32;
    let problem = Problem::new(&config);
    eprintln!("C3.2 cube: building the host FMM for its plan");
    let fmm = FmmBuilder::<f32>::new(3)
        .strategy(M2lStrategy::Rotation)
        .max_level(config.max_level)
        .max_points_per_leaf(config.max_points_per_leaf)
        .build(&problem.points, &problem.points, &comm)
        .expect("the host FMM builds on one rank");
    let views = levels(&fmm);
    let mut totals = Vec::new();
    for &p in &arguments.degrees {
        let tables = DegreeTables::new(p, cache.as_ref());
        let rows = measure(&mut device, &views, &tables);
        print_degree(p, &rows, &device);
        totals.push((p, rows));
    }
    print_summary(&totals);
    if !arguments.quick {
        let tables = DegreeTables::new(8, cache.as_ref());
        stage(&mut device, &views, &tables, arguments.threads);
        layouts(&mut device, views.last().expect("a V level"), &tables);
    }
}

//! The device M2M and L2L of `nd-fmm-kernels` (Phase 4 T8, C4.4): the grouped GEMM of
//! each level call of an FMM's plan, timed on its octant shapes, with its throughput
//! against the peak and against the Phase 0 GEMM spike, and the gather and the
//! accumulation timed apart. Prints Markdown on stdout and progress on stderr. Feature
//! `gpu` (with a backend).
//!
//! For each problem (the C3.2 cube: N = 10⁵, a uniform tree; the Plummer sphere of C3.3:
//! N = 10⁵, `max_level` 16, 64 points per leaf) and each p ∈ {3, 8, 16}, in f32 (or f64,
//! `--precision`):
//!
//! - the host `Fmm` is built once, for its plan; the multipoles and locals are seeded
//!   random coefficients (the timing does not depend on the values; every view is the
//!   plan's);
//! - per level call with a pair (M2M of the local and the global pass, L2L), on its octant
//!   batches as the device operator runs them (`nd_fmm_kernels::translate`):
//!   - **GEMM**, the hand-written kernel in the backend's default layout over the level's
//!     tile schedule (one launch), and, where the level's plan under `GemmPolicy::Auto`
//!     takes the library, the library GEMM on the padded batches (one launch);
//!   - **gather**, the level's input columns in batch order (one launch);
//!   - **level call**, the whole `grouped` call (gather, GEMM, reduction or scatter-add:
//!     three launches per chunk); the accumulation is the level call less the gather and
//!     the GEMM of the plan's engine (derived);
//! - each launch is checked first: the hand-written GEMM against the library's where both
//!   run (relative L2), so that a timed row is a correct one.
//!
//! Before each table the device runs half a second of other work (its clocks ramp down
//! while the host builds the FMM). Launches are queued between syncs, after a warm-up
//! launch and a sync (compilation excluded); each
//! figure is the median of 15 batches of at least 20 ms. GFLOP/s counts 2 n² flops per
//! column, n = (p + 1)²; % of peak against the device's peak in the run's precision from
//! `nd_fmm_validate::peaks` (Phase 4S T7): on the M3 Max 14.3 TFLOP/s in f32 (derived in the
//! spike report, not measured), on locust's GH200 the H100's datasheet figures without
//! tensor cores (67 TFLOP/s in f32, 34 TFLOP/s in f64); any other device has an unknown
//! peak and no percentage. The spike's figure is the Metal f32 GEMM of one table and B
//! columns (spikes/cubecl-gemm/results-m3max-0.11.md, CubeCL 0.11.0-pre.4) at the nearest
//! p of {4, 8, 12, 16} and the B of {10³, 10⁴, 10⁵} nearest the level's columns: its best
//! hand-written kernel, and the library CMMA; it is compared only on the M3 Max in f32
//! ("–" elsewhere). Nothing is asserted.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example translation_kernels \
//!     -- --device metal
//! ```
//!
//! `--device cpu` runs the same on the CubeCL CPU runtime (correctness backend; its
//! timings are not a Phase 4 target). `--quick` runs the C3.2 cube at p = 3 only. Metal
//! needs a process with GPU access (outside the macOS sandbox). It initialises MPI and
//! runs on one rank.
//!
//! Options of Phase 4S T7 (without them a run is as before):
//! - `--precision f32|f64` (default f32): the precision of the tables and coefficients;
//!   f64 is refused on a device without f64 (Metal);
//! - `--n N`: both problems at N points with the C3.3 tree settings (an adaptive tree,
//!   `max_level` 16, 64 points per leaf), the uniform cube in place of the C3.2 cube, so
//!   that the cube can run at N = 10⁶;
//! - `--degrees p,p,…`: the degrees (default 3, 8, 16);
//! - `--gemm rows,columns,per_unit`: the hand-written GEMM in the layout
//!   `GemmLayout::Cube { rows, columns, per_unit }` instead of the backend's default
//!   (`GemmLayout::default_for`); the header names it.
//!
//! On locust's GH200 (Phase 4S, by hand: `tools/gh200/remote.sh`, feature `cuda`),
//! `--device cuda` runs in f32 and f64; the library takes no shape there (device-path.md
//! F28), so every plan runs the hand-written kernel.

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_kernels::movement::gather_columns;
use nd_fmm_kernels::translate::{
    Accumulate, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands, Orientation, PlanSettings,
    Tables, TileSchedule, TranslationScratch, default_scratch_bytes, gemm, grouped, library,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_tables::{L2lTables, M2mTables};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};
use nd_fmm_validate::peaks::{self, Peaks};

/// The float types the example times, f32 and f64: on the device and in the tables.
trait Value: DeviceFloat + Stored {}

impl<T: DeviceFloat + Stored> Value for T {}

/// `x` in the precision of T, rounded to nearest.
fn value<T: Value>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// The device whose figures the spike measured (results-m3max-0.11.md).
const SPIKE_DEVICE: &str = "Apple M3 Max";

/// The spike's Metal f32 GFLOP/s (results-m3max-0.11.md): per p of [`SPIKE_DEGREES`] and B
/// of [`SPIKE_COLUMNS`], the best hand-written kernel (`tiled-smem`, `tiled-reg-cols`,
/// `tiled-reg-rows`) and the library `simple_cyclic_cmma` (`None`: rejected).
const SPIKE_DEGREES: [usize; 4] = [4, 8, 12, 16];
const SPIKE_COLUMNS: [usize; 3] = [1_000, 10_000, 100_000];
const SPIKE_HAND: [[f64; 3]; 4] = [
    [239.0, 1431.4, 3078.5],
    [625.4, 3585.1, 2921.2],
    [1426.1, 3903.3, 3081.7],
    [2145.5, 3572.2, 3309.8],
];
const SPIKE_LIBRARY: [[Option<f64>; 3]; 4] = [
    [None, None, None],
    [Some(2045.2), Some(3792.3), Some(4052.7)],
    [Some(3396.3), Some(4538.9), Some(3955.4)],
    [Some(3894.3), Some(4474.6), Some(4203.7)],
];

/// The command line.
struct Arguments {
    device: String,
    quick: bool,
    /// `--precision`: f32 by default.
    precision: Precision,
    /// `--n`: the problems at N points with the C3.3 tree settings.
    n: Option<usize>,
    /// `--degrees`: the degrees.
    degrees: Option<Vec<usize>>,
    /// `--gemm`: the hand-written GEMM's layout instead of the backend's default.
    gemm: Option<GemmLayout>,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: translation_kernels --device cpu|metal|cuda [--quick] [--precision \
             f32|f64] [--n N] [--degrees p,p,...] [--gemm rows,columns,per_unit]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        quick: false,
        precision: Precision::F32,
        n: None,
        degrees: None,
        gemm: None,
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--device", [value, tail @ ..]) => {
                parsed.device = value.clone();
                tail
            }
            ("--quick", _) => {
                parsed.quick = true;
                tail
            }
            ("--precision", [value, tail @ ..]) => {
                parsed.precision = match value.as_str() {
                    "f32" => Precision::F32,
                    "f64" => Precision::F64,
                    _ => usage(),
                };
                tail
            }
            ("--n", [value, tail @ ..]) => {
                let n: f64 = value.parse().unwrap_or_else(|_| usage());
                parsed.n = Some(n as usize);
                tail
            }
            ("--degrees", [value, tail @ ..]) => {
                parsed.degrees = Some(
                    value
                        .split(',')
                        .map(|p| p.parse().unwrap_or_else(|_| usage()))
                        .collect(),
                );
                tail
            }
            ("--gemm", [value, tail @ ..]) => {
                let v: Vec<u32> = value
                    .split(',')
                    .map(|x| x.parse().unwrap_or_else(|_| usage()))
                    .collect();
                let [rows, columns, per_unit] = v[..] else {
                    usage()
                };
                parsed.gemm = Some(GemmLayout::Cube {
                    rows,
                    columns,
                    per_unit,
                });
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty()
        || parsed.n == Some(0)
        || parsed.degrees.as_ref().is_some_and(Vec::is_empty)
    {
        usage();
    }
    parsed
}

/// A grouped view of the plan, copied.
struct View {
    name: &'static str,
    level: usize,
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u8>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
    /// The level of the input columns.
    input_level: usize,
    m2m: bool,
}

impl View {
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
}

/// One timed level call.
struct Row {
    view: String,
    level: usize,
    pairs: usize,
    k_max: usize,
    /// Seconds per launch: the hand-written GEMM, the library GEMM (if it took the
    /// shape), the gather, the whole level call.
    hand: f64,
    library: Option<f64>,
    gather: f64,
    call: f64,
    /// The plan's GEMM and chunks.
    gemm: Gemm,
    chunks: usize,
    /// Relative L2 of the hand-written product from the library's, where both ran.
    agreement: Option<f64>,
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
    if !device.supports(arguments.precision) {
        eprintln!(
            "--precision {}: {} does no arithmetic in it",
            precision_name(arguments.precision),
            device.info()
        );
        std::process::exit(1);
    }
    if let Some(layout) = arguments.gemm
        && let Err(error) = layout.check(device.info())
    {
        eprintln!("--gemm: {error}");
        std::process::exit(1);
    }
    match arguments.precision {
        Precision::F32 => run::<f32>(&mut device, &arguments),
        Precision::F64 => run::<f64>(&mut device, &arguments),
    }
}

/// The name of a precision.
fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F32 => "f32",
        Precision::F64 => "f64",
    }
}

/// The precision of T in the peak table.
fn peak_precision<T: Value>() -> peaks::Precision {
    match T::FLOAT {
        Precision::F32 => peaks::Precision::F32,
        Precision::F64 => peaks::Precision::F64,
    }
}

/// The whole run in the precision of T.
fn run<T: Value>(device: &mut Device, arguments: &Arguments) {
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let comm = universe.world();
    print_header::<T>(device, arguments);
    let mut problems = match arguments.n {
        None => vec![("C3.2 cube", Config::C32)],
        Some(n) => vec![(
            "uniform cube",
            Config {
                n,
                ..Config::c33(Distribution::Cube)
            },
        )],
    };
    if !arguments.quick {
        problems.push((
            "Plummer sphere",
            Config {
                n: arguments.n.unwrap_or(Config::C32.n),
                ..Config::c33(Distribution::Plummer)
            },
        ));
    }
    let degrees: Vec<usize> = match (&arguments.degrees, arguments.quick) {
        (Some(degrees), _) => degrees.clone(),
        (None, true) => vec![3],
        (None, false) => vec![3, 8, 16],
    };
    for (name, config) in problems {
        let problem = Problem::new(&config);
        for &p in &degrees {
            eprintln!("{name}, p = {p}: building the host FMM");
            // The plan only: rotation tables are the cheapest to build at any p.
            let fmm = FmmBuilder::<f32>::new(p)
                .strategy(M2lStrategy::Rotation)
                .max_level(config.max_level)
                .max_points_per_leaf(config.max_points_per_leaf)
                .build(&problem.points, &problem.points, &comm)
                .expect("the host FMM builds on one rank");
            let rows = measure::<T>(device, (name, p), &fmm, arguments.gemm);
            print_table::<T>(name, p, &rows, device, arguments.gemm);
        }
    }
}

fn print_header<T: Value>(device: &Device, arguments: &Arguments) {
    println!("# The device M2M and L2L per level (Phase 4 T8, C4.4)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {} |", device.info());
    println!(
        "| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a \
         warm-up launch and a sync first (compilation excluded) |"
    );
    let peak = match Peaks::of_info(device.info()) {
        Some(p) => match p.gflops(peak_precision::<T>()) {
            Some(g) => format!("{g} GFLOP/s ({})", p.source(peak_precision::<T>())),
            None => format!("unknown peak ({})", p.source(peak_precision::<T>())),
        },
        None => format!(
            "unknown peak ({} is not in `nd_fmm_validate::peaks`)",
            device.info().name
        ),
    };
    println!("| peak | {peak}; GFLOP/s counts 2 n² per column |");
    match (T::FLOAT, device.backend()) {
        (Precision::F32, BackendKind::Cpu | BackendKind::Metal) => println!(
            "| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |"
        ),
        (precision, backend) => {
            println!("| precision | {} on {backend} |", precision_name(precision))
        }
    }
    if let Some(n) = arguments.n {
        println!(
            "| problems | N = {n}, the C3.3 tree settings (`max_level` 16, 64 points per leaf) \
             |"
        );
    }
    if let Some(layout) = arguments.gemm {
        println!("| hand-written GEMM | {layout} (`--gemm`) at every p |");
    }
    println!();
}

/// The views of every level call of M2M and L2L with a pair.
fn views(fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>) -> Vec<View> {
    let plan = fmm.plan();
    let mut views = Vec::new();
    for (level, lists) in plan.levels().iter().enumerate() {
        for (name, view, m2m) in [
            ("M2M local", lists.m2m_local(), true),
            ("M2M global", lists.m2m_global(), true),
            ("L2L", lists.l2l(), false),
        ] {
            if view.is_empty() {
                continue;
            }
            views.push(View {
                name,
                level,
                row_offsets: view.row_offsets().to_vec(),
                sources: view.sources().to_vec(),
                groups: view.groups().to_vec(),
                batch_offsets: view.batch_offsets().to_vec(),
                batch_targets: view.batch_targets().to_vec(),
                batch_sources: view.batch_sources().to_vec(),
                input_level: if m2m { level + 1 } else { level - 1 },
                m2m,
            });
        }
    }
    views
}

fn measure<T: Value>(
    device: &mut Device,
    (problem, p): (&'static str, usize),
    fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>,
    chosen: Option<GemmLayout>,
) -> Vec<Row> {
    let n = (p + 1) * (p + 1);
    let index = fmm.plan().index();
    let mut rng = SplitMix64::new(0x7e_8001);
    let layout = chosen.unwrap_or_else(|| GemmLayout::default_for(device.info(), n));
    let library_copy = PlanSettings {
        n,
        layout,
        policy: GemmPolicy::Auto,
        budget: default_scratch_bytes(device.info()),
        orientation: Orientation::BoxMajor,
    }
    .library_candidate(device.backend(), T::FLOAT);
    let m2m_tables = Tables::upload(
        device,
        M2mTables::<T>::build(p).matrices().as_slice(),
        n,
        library_copy,
    )
    .unwrap();
    let l2l_tables = Tables::upload(
        device,
        L2lTables::<T>::build(p).matrices().as_slice(),
        n,
        library_copy,
    )
    .unwrap();
    warm_up(device);
    let mut rows = Vec::new();
    for view in views(fmm) {
        eprintln!("{problem}, p = {p}: {} on level {}", view.name, view.level);
        let tables = if view.m2m { &m2m_tables } else { &l2l_tables };
        let (inputs, outputs) = (index.len(view.input_level), index.len(view.level));
        let store: Vec<T> = (0..(inputs + outputs) * n)
            .map(|i| value::<T>(rng.range(-1.0, 1.0)) / value::<T>(1.0 + (i % n) as f64))
            .collect();
        let mut buffer = device.upload(&store).unwrap();
        let device_view = GroupedView::upload(device, &view.arrays(), inputs).unwrap();
        let settings = PlanSettings {
            n,
            layout,
            policy: GemmPolicy::Auto,
            budget: default_scratch_bytes(device.info()),
            orientation: Orientation::BoxMajor,
        };
        let size = settings.size(device.backend(), T::FLOAT, &view.batch_offsets);
        let mut scratch = TranslationScratch::<T>::new(device, size.columns * n).unwrap();
        let plan = GroupedPlan::new(
            device,
            &view.arrays(),
            inputs,
            &settings,
            tables,
            &mut scratch,
        )
        .unwrap();
        let pairs = view.batch_sources.len();
        let k_max = view
            .batch_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);

        // The gather and the hand-written GEMM over the level's batches in one chunk.
        let input = buffer.slice(..inputs * n);
        let sources = device.upload_indices(&view.batch_sources).unwrap();
        let mut x = device.alloc::<T>(pairs * n).unwrap();
        let mut y = device.alloc::<T>(pairs * n).unwrap();
        let run_gather = |device: &mut Device, x: &mut DeviceBuffer<T>| {
            gather_columns(device, n, input, sources.as_slice(), x.as_slice_mut()).unwrap();
        };
        run_gather(device, &mut x);
        device.sync().unwrap();
        let gather = median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                run_gather(device, &mut x);
            }
            device.sync().unwrap();
            start.elapsed()
        });
        let schedule = TileSchedule::new(device, layout, &view.batch_offsets).unwrap();
        let run_hand = |device: &mut Device, y: &mut DeviceBuffer<T>| {
            gemm(
                device,
                layout,
                n,
                tables.compact(),
                &schedule,
                x.as_slice(),
                y.as_slice_mut(),
            )
            .unwrap();
        };
        run_hand(device, &mut y);
        device.sync().unwrap();
        let hand = median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                run_hand(device, &mut y);
            }
            device.sync().unwrap();
            start.elapsed()
        });

        // The library on the padded batches, where the plan took it.
        let (library_time, agreement) = if plan.gemm() == Gemm::Library {
            let groups = view.batch_offsets.len() - 1;
            let width = groups * k_max;
            let mut padded = vec![view.batch_sources[0]; width];
            for (g, w) in view.batch_offsets.windows(2).enumerate() {
                let batch = &view.batch_sources[w[0] as usize..w[1] as usize];
                padded[g * k_max..g * k_max + batch.len()].copy_from_slice(batch);
            }
            let padded = device.upload_indices(&padded).unwrap();
            let mut xp = device.alloc::<T>(width * n).unwrap();
            let mut yp = device.alloc::<T>(width * n).unwrap();
            gather_columns(device, n, input, padded.as_slice(), xp.as_slice_mut()).unwrap();
            let run_library = |device: &mut Device, yp: &mut DeviceBuffer<T>| {
                library(
                    device,
                    (groups, k_max, n),
                    tables.compact(),
                    xp.as_slice(),
                    yp.as_slice_mut(),
                )
                .unwrap();
            };
            run_library(device, &mut yp);
            // Compare the products column by column with the hand-written ones.
            let mut got = vec![value::<T>(0.0); width * n];
            device.download(yp.as_slice(), &mut got).unwrap();
            let mut want = vec![value::<T>(0.0); pairs * n];
            device.download(y.as_slice(), &mut want).unwrap();
            let (mut d2, mut r2) = (0.0f64, 0.0f64);
            for (g, w) in view.batch_offsets.windows(2).enumerate() {
                for (j, k) in (w[0] as usize..w[1] as usize).enumerate() {
                    for i in 0..n {
                        let a = RealScalar::to_f64(got[(g * k_max + j) * n + i]);
                        let b = RealScalar::to_f64(want[k * n + i]);
                        d2 += (a - b) * (a - b);
                        r2 += b * b;
                    }
                }
            }
            let seconds = median_time_per_call(|calls| {
                let start = Instant::now();
                for _ in 0..calls {
                    run_library(device, &mut yp);
                }
                device.sync().unwrap();
                start.elapsed()
            });
            (Some(seconds), Some((d2 / r2.max(f64::MIN_POSITIVE)).sqrt()))
        } else {
            (None, None)
        };

        // The whole level call, as the device operator runs it.
        let accumulate = if view.m2m {
            Accumulate::Rows
        } else {
            Accumulate::Scatter
        };
        let mut run_call = |device: &mut Device, buffer: &mut DeviceBuffer<T>| {
            grouped(
                device,
                &plan,
                &device_view,
                accumulate,
                tables,
                Operands::Shared {
                    buffer,
                    input: 0..inputs * n,
                    output: inputs * n..(inputs + outputs) * n,
                },
                &mut scratch,
            )
            .unwrap();
        };
        run_call(device, &mut buffer);
        device.sync().unwrap();
        let call = median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                run_call(device, &mut buffer);
            }
            device.sync().unwrap();
            start.elapsed()
        });
        rows.push(Row {
            view: view.name.to_owned(),
            level: view.level,
            pairs,
            k_max,
            hand,
            library: library_time,
            gather,
            call,
            gemm: plan.gemm(),
            chunks: plan.nchunks(),
            agreement,
        });
    }
    rows
}

/// Keeps the device busy for half a second, so that the first timed launch does not meet a
/// GPU clocked down while the host built the FMM (the first row was 10–30 times slower
/// than the same shape later without it).
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

/// The spike's cell nearest (p, columns): (p, B, best hand-written, library).
fn spike(p: usize, columns: usize) -> (usize, usize, f64, Option<f64>) {
    let i = (0..SPIKE_DEGREES.len())
        .min_by_key(|&i| SPIKE_DEGREES[i].abs_diff(p))
        .unwrap();
    let ln = (columns.max(1) as f64).ln();
    let j = (0..SPIKE_COLUMNS.len())
        .min_by(|&a, &b| {
            let da = ((SPIKE_COLUMNS[a] as f64).ln() - ln).abs();
            let db = ((SPIKE_COLUMNS[b] as f64).ln() - ln).abs();
            da.total_cmp(&db)
        })
        .unwrap();
    (
        SPIKE_DEGREES[i],
        SPIKE_COLUMNS[j],
        SPIKE_HAND[i][j],
        SPIKE_LIBRARY[i][j],
    )
}

fn print_table<T: Value>(
    problem: &str,
    p: usize,
    rows: &[Row],
    device: &Device,
    chosen: Option<GemmLayout>,
) {
    let n = (p + 1) * (p + 1);
    let peak = Peaks::of_info(device.info()).and_then(|p| p.gflops(peak_precision::<T>()));
    let percent = |g: f64| peak.map_or("–".to_owned(), |peak| format!("{:.1}", 100.0 * g / peak));
    let spike_applies = device.info().name == SPIKE_DEVICE && T::FLOAT == Precision::F32;
    println!("## {problem}, p = {p} (n = {n}), {}", device.backend());
    println!();
    println!(
        "| view | level | pairs | k_max | plan's GEMM (chunks) | hand-written GEMM µs | \
         GFLOP/s | % peak | library GEMM µs | GFLOP/s | % peak | spike (p, B): hand / \
         library GFLOP/s | gather µs | level call µs | accumulate µs (derived) | gather + \
         accumulate share |"
    );
    println!(
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: \
         | ---: | ---: | ---: |"
    );
    let mut totals = (0.0, 0.0, 0.0, 0.0);
    for r in rows {
        let flops = 2.0 * (n * n * r.pairs) as f64;
        let gflops = |seconds: f64| flops / seconds / 1e9;
        let (sp, sb, shand, slib) = spike(p, r.pairs);
        let spike_cell = if spike_applies {
            format!(
                "({sp}, {sb}): {shand:.0} / {}",
                slib.map_or("rejected".to_owned(), |g| format!("{g:.0}"))
            )
        } else {
            "–".to_owned()
        };
        let engine = match (r.gemm, r.library) {
            (Gemm::Library, Some(t)) => t,
            _ => r.hand,
        };
        let accumulate = (r.call - r.gather - engine).max(0.0);
        totals.0 += r.call;
        totals.1 += r.gather;
        totals.2 += engine;
        totals.3 += accumulate;
        println!(
            "| {} | {} | {} | {} | {} ({}) | {:.1} | {:.0} | {} | {} | {} | {} | {spike_cell} | \
             {:.1} | {:.1} | {:.1} | {:.0}% |",
            r.view,
            r.level,
            r.pairs,
            r.k_max,
            match r.gemm {
                Gemm::Library => "library".to_owned(),
                Gemm::HandWritten(_) => "hand-written".to_owned(),
            },
            r.chunks,
            r.hand * 1e6,
            gflops(r.hand),
            percent(gflops(r.hand)),
            r.library
                .map_or("–".to_owned(), |t| format!("{:.1}", t * 1e6)),
            r.library
                .map_or("–".to_owned(), |t| format!("{:.0}", gflops(t))),
            r.library.map_or("–".to_owned(), |t| percent(gflops(t))),
            r.gather * 1e6,
            r.call * 1e6,
            accumulate * 1e6,
            100.0 * (r.gather + accumulate) / r.call,
        );
        if let Some(a) = r.agreement {
            eprintln!(
                "  {} level {}: hand-written against library, relative L2 {a:.2e}",
                r.view, r.level
            );
        }
    }
    println!();
    println!(
        "Totals over the level calls: {:.1} µs per evaluation's M2M and L2L ({} calls), of \
         which GEMM {:.1} µs, gather {:.1} µs, accumulation {:.1} µs (gather and \
         accumulation {:.0}%). The hand-written layout: {}.",
        totals.0 * 1e6,
        rows.len(),
        totals.2 * 1e6,
        totals.1 * 1e6,
        totals.3 * 1e6,
        100.0 * (totals.1 + totals.3) / totals.0.max(f64::MIN_POSITIVE),
        chosen.unwrap_or_else(|| GemmLayout::default_for(device.info(), n))
    );
    println!();
}

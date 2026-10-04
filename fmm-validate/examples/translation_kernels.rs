//! The device M2M and L2L of `nd-fmm-kernels` (Phase 4 T8, C4.4): the grouped GEMM of
//! each level call of an FMM's plan, timed on its octant shapes, with its throughput
//! against the peak and against the Phase 0 GEMM spike, and the gather and the
//! accumulation timed apart. Prints Markdown on stdout and progress on stderr. Feature
//! `gpu` (with a backend).
//!
//! For each problem (the C3.2 cube: N = 10⁵, a uniform tree; the Plummer sphere of C3.3:
//! N = 10⁵, `max_level` 16, 64 points per leaf) and each p ∈ {3, 8, 16}, in f32:
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
//! column, n = (p + 1)²; % of peak against 14.3 TFLOP/s (the M3 Max GPU in f32, derived in
//! the spike report, not measured). The spike's figure is the Metal f32 GEMM of one table
//! and B columns (spikes/cubecl-gemm/results-m3max-0.11.md, CubeCL 0.11.0-pre.4) at the
//! nearest p of {4, 8, 12, 16} and the B of {10³, 10⁴, 10⁵} nearest the level's columns:
//! its best hand-written kernel, and the library CMMA. Nothing is asserted. f64 is not
//! timed on a GPU (Metal has none) and CUDA is type-checked only.
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

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_kernels::movement::gather_columns;
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    PlanSettings, Tables, TileSchedule, TranslationScratch, gemm, grouped, library,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer};
use nd_fmm_tables::{L2lTables, M2mTables};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};

/// The f32 peak of the M3 Max GPU (spikes/cubecl-gemm/SPIKE_REPORT.md, derived).
const PEAK_GFLOPS: f64 = 14_300.0;

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
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!("usage: translation_kernels --device cpu|metal|cuda [--quick]; got {args:?}");
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        quick: false,
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
            _ => usage(),
        };
    }
    if parsed.device.is_empty() {
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
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let comm = universe.world();
    print_header(&device);
    let mut problems = vec![("C3.2 cube", Config::C32)];
    if !arguments.quick {
        problems.push(("Plummer sphere", Config::c33(Distribution::Plummer)));
    }
    let degrees: &[usize] = if arguments.quick { &[3] } else { &[3, 8, 16] };
    for (name, config) in problems {
        let problem = Problem::new(&config);
        for &p in degrees {
            eprintln!("{name}, p = {p}: building the host FMM");
            // The plan only: rotation tables are the cheapest to build at any p.
            let fmm = FmmBuilder::<f32>::new(p)
                .strategy(M2lStrategy::Rotation)
                .max_level(config.max_level)
                .max_points_per_leaf(config.max_points_per_leaf)
                .build(&problem.points, &problem.points, &comm)
                .expect("the host FMM builds on one rank");
            let rows = measure(&mut device, (name, p), &fmm);
            print_table(name, p, &rows, &device);
        }
    }
}

fn print_header(device: &Device) {
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
    println!(
        "| peak | {PEAK_GFLOPS} GFLOP/s (M3 Max GPU f32, derived in the spike report, not \
         measured); GFLOP/s counts 2 n² per column |"
    );
    println!("| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |");
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

fn measure(
    device: &mut Device,
    (problem, p): (&'static str, usize),
    fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>,
) -> Vec<Row> {
    let n = (p + 1) * (p + 1);
    let index = fmm.plan().index();
    let mut rng = SplitMix64::new(0x7e_8001);
    let layout = GemmLayout::default_for(device.info(), n);
    let library_copy = PlanSettings {
        n,
        layout,
        policy: GemmPolicy::Auto,
        budget: DEFAULT_SCRATCH_BYTES,
    }
    .library_candidate(device.backend(), nd_fmm_kernels::Precision::F32);
    let m2m_tables = Tables::upload(
        device,
        M2mTables::<f32>::build(p).matrices().as_slice(),
        n,
        library_copy,
    )
    .unwrap();
    let l2l_tables = Tables::upload(
        device,
        L2lTables::<f32>::build(p).matrices().as_slice(),
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
        let store: Vec<f32> = (0..(inputs + outputs) * n)
            .map(|i| rng.range(-1.0, 1.0) as f32 / (1.0 + (i % n) as f32))
            .collect();
        let mut buffer = device.upload(&store).unwrap();
        let device_view = GroupedView::upload(device, &view.arrays(), inputs).unwrap();
        let settings = PlanSettings {
            n,
            layout,
            policy: GemmPolicy::Auto,
            budget: DEFAULT_SCRATCH_BYTES,
        };
        let size = settings.size(
            device.backend(),
            nd_fmm_kernels::Precision::F32,
            &view.batch_offsets,
        );
        let mut scratch = TranslationScratch::<f32>::new(device, size.columns * n).unwrap();
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
        let mut x = device.alloc::<f32>(pairs * n).unwrap();
        let mut y = device.alloc::<f32>(pairs * n).unwrap();
        let run_gather = |device: &mut Device, x: &mut DeviceBuffer<f32>| {
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
        let run_hand = |device: &mut Device, y: &mut DeviceBuffer<f32>| {
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
            let mut xp = device.alloc::<f32>(width * n).unwrap();
            let mut yp = device.alloc::<f32>(width * n).unwrap();
            gather_columns(device, n, input, padded.as_slice(), xp.as_slice_mut()).unwrap();
            let run_library = |device: &mut Device, yp: &mut DeviceBuffer<f32>| {
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
            let mut got = vec![0.0f32; width * n];
            device.download(yp.as_slice(), &mut got).unwrap();
            let mut want = vec![0.0f32; pairs * n];
            device.download(y.as_slice(), &mut want).unwrap();
            let (mut d2, mut r2) = (0.0f64, 0.0f64);
            for (g, w) in view.batch_offsets.windows(2).enumerate() {
                for (j, k) in (w[0] as usize..w[1] as usize).enumerate() {
                    for i in 0..n {
                        let a = f64::from(got[(g * k_max + j) * n + i]);
                        let b = f64::from(want[k * n + i]);
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
        let mut run_call = |device: &mut Device, buffer: &mut DeviceBuffer<f32>| {
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

fn print_table(problem: &str, p: usize, rows: &[Row], device: &Device) {
    let n = (p + 1) * (p + 1);
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
            "| {} | {} | {} | {} | {} ({}) | {:.1} | {:.0} | {:.1} | {} | {} | {} | ({sp}, {sb}): \
             {shand:.0} / {} | {:.1} | {:.1} | {:.1} | {:.0}% |",
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
            100.0 * gflops(r.hand) / PEAK_GFLOPS,
            r.library
                .map_or("–".to_owned(), |t| format!("{:.1}", t * 1e6)),
            r.library
                .map_or("–".to_owned(), |t| format!("{:.0}", gflops(t))),
            r.library.map_or("–".to_owned(), |t| format!(
                "{:.1}",
                100.0 * gflops(t) / PEAK_GFLOPS
            )),
            slib.map_or("rejected".to_owned(), |g| format!("{g:.0}")),
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
        GemmLayout::default_for(device.info(), n)
    );
    println!();
}

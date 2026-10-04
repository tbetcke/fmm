//! The device dense M2L of `nd-fmm-kernels` (Phase 4 T9, C4.5): the grouped translation of
//! every V-list level of an FMM's plan, timed stage by stage, its GEMM against the peak and
//! against the Phase 0 GEMM spike, against the per-offset structure (A), and the whole M2L
//! stage against the host operator; then the GEMM alone on the spike's shapes (the C4.5
//! gate) and the level call under several scratch budgets. Prints Markdown on stdout and
//! progress on stderr. Feature `gpu` (with a backend).
//!
//! For each problem (the C3.2 cube: N = 10⁵, a uniform level-4 tree; the Plummer sphere of
//! C3.3: N = 10⁵, `max_level` 16, 64 points per leaf), in f32:
//!
//! - the host `Fmm` is built once, for its plan (the V views do not depend on p); the
//!   multipoles are seeded random coefficients (the timing does not depend on the values);
//! - for each p ∈ {3, 8} (the f32 FMM degrees) and p ∈ {12, 16} (the kernel only, as model
//!   input for f64), per level with a V pair, the level call as the device operator runs it
//!   under `DeviceGemm::Library` (`nd_fmm_kernels::translate::grouped` with the 316 dense
//!   tables of `M2lTables`, the GEMM of `GemmPolicy::Auto`: the library CMMA at p ≥ 8 where
//!   it takes the shape, the hand-written kernel otherwise, in the backend's default
//!   layout); the hand-written rows are the device default, `DeviceGemm::Auto`:
//!   - **GEMM**, **gather** and **accumulation** (the row-ordered reduction), each over
//!     every chunk of the level (`grouped_stage`, one launch per chunk), and the whole
//!     **level call** (three launches per chunk);
//!   - where the plan took the library, the hand-written GEMM on the same batches too;
//!   - structure (A), one gather, GEMM and scatter-add per offset (`per_group`), with the
//!     plan's GEMM: its time and launches;
//!   - at p ≤ 8, the host operator's M2L of the level (`LaplaceOperator::m2l_pair`, the
//!     body of its level call, with the dense f32 tables) on one thread and on `--threads`
//!     scoped threads (default 12, the performance cores), each a contiguous share of the
//!     level's boxes, with every BLAS thread variable set to 1 by the launcher;
//!   - first, the level call from zero locals is compared with the host's (p ≤ 8) or, at
//!     p ≥ 12, the library's with the hand-written kernel's (relative L2 over the level),
//!     so that a timed row is a correct one;
//! - **the gate** (C4.5): the GEMM alone on the spike's shapes, one table and B columns,
//!   B ∈ {10³, 10⁴, 10⁵}, at p ∈ {4, 8, 12, 16}: the GEMM the rule picks (the library at
//!   p ≥ 8, the hand-written kernel at p = 4) against the spike's figure for that GEMM at
//!   the same (p, B), met at 80% or more; the hand-written kernel at p ≥ 8 for reference;
//! - **the scratch budget**: the level call of the cube's deepest V level at p = 8 under
//!   budgets of 8 to 512 MB (chunks and time);
//! - **the orientation** (Phase 4 T12, README decision 12): for both problems at
//!   p ∈ {3, 6, 8} (the f32 FMM degrees) and p ∈ {12, 16}, per V level, the whole level
//!   call box-major (the layout of device-path.md §6.4) and coefficient-major (the spike's
//!   orientation, `Orientation::CoefficientMajor`), with the hand-written GEMM and, where
//!   it takes the level's shapes, the library; the coefficient-major results are checked
//!   against the box-major ones first (relative L2; the hand-written kernel's are equal
//!   bit for bit by construction). `--orientation` runs this table alone.
//!
//! Timing: before each table the device runs half a second of other work (its clocks ramp
//! down while the host builds); launches are queued between syncs, after a warm-up launch
//! and a sync (compilation excluded); each figure is the median of 15 batches of at least
//! 20 ms. GFLOP/s counts 2 n² flops per pair (useful flops; the library's padding columns
//! are not counted), n = (p + 1)²; % of peak against 14.3 TFLOP/s (the M3 Max GPU in f32,
//! derived in the spike report, not measured). The spike's figures are the Metal f32 GEMM
//! of one table and B columns (spikes/cubecl-gemm/results-m3max-0.11.md, CubeCL
//! 0.11.0-pre.4): the best hand-written kernel and the library CMMA. Per level, the spike
//! cell is the one at the nearest p of {4, 8, 12, 16} and the B nearest the level's mean
//! columns per offset (log scale). Nothing is asserted. f64 is not timed on a GPU (Metal
//! has none) and CUDA is type-checked only.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example m2l_kernels -- \
//!     --device metal [--table-cache DIR]
//! ```
//!
//! `--device cpu` runs the same on the CubeCL CPU runtime (correctness backend; its
//! timings are not a Phase 4 target; use `--quick` there). `--quick` runs the C3.2 cube
//! at p = 3 only and skips the gate and the budget sweep. `--table-cache DIR` loads and
//! stores the tables there (`nd_fmm_tables::TableCache`). Metal needs a process with GPU
//! access (outside the macOS sandbox). It initialises MPI and runs on one rank.

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables as HostTables};
use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    Orientation, PerGroupPlan, PlanSettings, Stage, Tables, TileSchedule, TranslationScratch, gemm,
    grouped, grouped_stage, library, per_group,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_tables::{M2lTables, TableCache};
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

/// The C4.5 gate: at least this share of the spike's throughput.
const GATE: f64 = 0.8;

/// The command line.
struct Arguments {
    device: String,
    threads: usize,
    quick: bool,
    orientation_only: bool,
    table_cache: Option<String>,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: m2l_kernels --device cpu|metal|cuda [--threads n] [--quick] \
             [--orientation] [--table-cache DIR]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        threads: 12,
        quick: false,
        orientation_only: false,
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
            ("--table-cache", [value, tail @ ..]) => {
                parsed.table_cache = Some(value.clone());
                tail
            }
            ("--quick", _) => {
                parsed.quick = true;
                tail
            }
            ("--orientation", _) => {
                parsed.orientation_only = true;
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty() || parsed.threads == 0 {
        usage();
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

    /// The offsets with a pair, and the mean and largest columns per such offset.
    fn batches(&self) -> (usize, f64, usize) {
        let k: Vec<usize> = self
            .batch_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .filter(|&k| k > 0)
            .collect();
        let mean = k.iter().sum::<usize>() as f64 / k.len().max(1) as f64;
        (k.len(), mean, k.iter().copied().max().unwrap_or(0))
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

/// One timed level.
struct Row {
    level: usize,
    boxes: usize,
    pairs: usize,
    offsets: usize,
    k_mean: f64,
    k_max: usize,
    gemm: Gemm,
    chunks: usize,
    gemm_columns: usize,
    rejection: Option<String>,
    /// Seconds per launch sequence: the plan's GEMM, gather and accumulation over every
    /// chunk, the whole level call.
    gemm_time: f64,
    gather: f64,
    accumulate: f64,
    call: f64,
    /// The hand-written GEMM where the plan took the library.
    hand: Option<f64>,
    /// Structure (A): seconds and launches.
    per_offset: f64,
    per_offset_launches: usize,
    per_offset_gemm: Gemm,
    /// The host at one and many threads (p ≤ 8).
    host: Option<(f64, f64)>,
    /// Relative L2 of the device result against the check (module documentation).
    difference: f64,
    check: &'static str,
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
    let mut problems = vec![("C3.2 cube", Config::C32)];
    if !arguments.quick {
        problems.push(("Plummer sphere", Config::c33(Distribution::Plummer)));
    }
    let degrees: &[usize] = if arguments.quick {
        &[3]
    } else {
        &[3, 8, 12, 16]
    };
    let mut budget_levels = None;
    let mut orientation_views = Vec::new();
    for (name, config) in problems {
        let problem = Problem::new(&config);
        eprintln!("{name}: building the host FMM for its plan");
        // The plan only: rotation tables are the cheapest to build.
        let fmm = FmmBuilder::<f32>::new(3)
            .strategy(M2lStrategy::Rotation)
            .max_level(config.max_level)
            .max_points_per_leaf(config.max_points_per_leaf)
            .build(&problem.points, &problem.points, &comm)
            .expect("the host FMM builds on one rank");
        let views = levels(&fmm);
        if arguments.orientation_only {
            orientation_views.push((name, views));
            continue;
        }
        let mut totals = Vec::new();
        for &p in degrees {
            let (host_tables, m2l) = tables(p, cache.as_ref());
            let rows = measure(
                &mut device,
                (name, p),
                &views,
                &m2l,
                host_tables.map(|t| LaplaceOperator::new(t, false, 64)),
                arguments.threads,
            );
            print_levels(name, p, &rows, &device, arguments.threads);
            totals.push((p, rows));
        }
        print_totals(name, &totals, arguments.threads);
        if budget_levels.is_none() {
            budget_levels = Some(views);
        }
    }
    if arguments.orientation_only {
        let degrees: &[usize] = if arguments.quick {
            &[3]
        } else {
            &[3, 6, 8, 12, 16]
        };
        orientations(&mut device, &orientation_views, degrees, cache.as_ref());
        return;
    }
    if !arguments.quick {
        gate(&mut device);
        let views = budget_levels.as_ref().expect("the cube ran");
        let deepest = views.last().expect("a V level");
        let (_, m2l) = tables(8, cache.as_ref());
        budgets(&mut device, deepest, &m2l);
        let mut cube = budget_levels;
        let plummer = {
            let config = Config::c33(Distribution::Plummer);
            let problem = Problem::new(&config);
            let fmm = FmmBuilder::<f32>::new(3)
                .strategy(M2lStrategy::Rotation)
                .max_level(config.max_level)
                .max_points_per_leaf(config.max_points_per_leaf)
                .build(&problem.points, &problem.points, &comm)
                .expect("the host FMM builds on one rank");
            levels(&fmm)
        };
        let views = vec![
            ("C3.2 cube", cube.take().expect("the cube ran")),
            ("Plummer sphere", plummer),
        ];
        orientations(&mut device, &views, &[3, 6, 8, 12, 16], cache.as_ref());
    }
}

/// The level call of every V level of `views` box-major and coefficient-major, with the
/// hand-written GEMM and the library (module documentation, "the orientation").
fn orientations(
    device: &mut Device,
    views: &[(&str, Vec<Level>)],
    degrees: &[usize],
    cache: Option<&TableCache>,
) {
    println!("## The orientation of X and Y (T12): box-major against coefficient-major");
    println!();
    println!(
        "Whole level call (gather, GEMM, reduction per chunk), µs; `cm/bm` < 1 means the \
         coefficient-major layout is faster. Library columns where the library takes the \
         level's shapes in that orientation (f32, p ≥ 8, a GPU)."
    );
    println!();
    println!(
        "| problem | p | level | pairs | hand bm | hand cm | cm/bm | library bm | library cm \
         | cm/bm | best |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
    for &p in degrees {
        let n = (p + 1) * (p + 1);
        let (_, m2l) = tables(p, cache);
        let auto = settings(device, n, GemmPolicy::Auto, DEFAULT_SCRATCH_BYTES);
        let library_copy = auto.library_candidate(device.backend(), Precision::F32);
        let tables = Tables::upload(device, &m2l, n, library_copy).unwrap();
        warm_up(device);
        for (problem, levels) in views {
            for level in levels {
                eprintln!("orientation: {problem}, p = {p}, level {}", level.level);
                let boxes = level.boxes();
                let mut rng = SplitMix64::new(0x7e_9300 + p as u64);
                let multipoles_host: Vec<f32> = (0..boxes * n)
                    .map(|_| rng.range(-1.0, 1.0) as f32)
                    .collect();
                let multipoles = device.upload(&multipoles_host).unwrap();
                let mut locals = device.alloc::<f32>(boxes * n).unwrap();
                let view = GroupedView::upload(device, &level.arrays(), boxes).unwrap();
                // (time, result) per (policy, orientation); None where the library refused.
                let mut cells: Vec<Option<(f64, Vec<f32>)>> = Vec::new();
                for policy in [GemmPolicy::HandWritten, GemmPolicy::Auto] {
                    for orientation in [Orientation::BoxMajor, Orientation::CoefficientMajor] {
                        let s = PlanSettings {
                            orientation,
                            ..settings(device, n, policy, DEFAULT_SCRATCH_BYTES)
                        };
                        let (plan, mut scratch) = plan_of(device, level, &s, &tables);
                        if policy == GemmPolicy::Auto && plan.gemm() != Gemm::Library {
                            cells.push(None);
                            continue;
                        }
                        let mut run = |d: &mut Device, locals: &mut DeviceBuffer<f32>| {
                            grouped(
                                d,
                                &plan,
                                &view,
                                Accumulate::Rows,
                                &tables,
                                Operands::Separate {
                                    input: multipoles.as_slice(),
                                    output: locals.as_slice_mut(),
                                },
                                &mut scratch,
                            )
                            .unwrap();
                        };
                        nd_fmm_kernels::movement::zero(device, locals.as_slice_mut()).unwrap();
                        run(device, &mut locals);
                        let mut result = vec![0.0f32; boxes * n];
                        device.download(locals.as_slice(), &mut result).unwrap();
                        let seconds = time(device, |d| run(d, &mut locals));
                        cells.push(Some((seconds, result)));
                    }
                }
                let reference = &cells[0].as_ref().expect("the hand-written kernel runs").1;
                let mut worst = 0.0f64;
                for cell in cells.iter().flatten() {
                    worst = worst.max(relative_l2(&cell.1, reference));
                }
                assert!(
                    worst < 1e-5,
                    "{problem}, p = {p}, level {}: the orientations differ by {worst:e}",
                    level.level
                );
                let us = |c: &Option<(f64, Vec<f32>)>| {
                    c.as_ref()
                        .map_or("–".to_owned(), |(t, _)| format!("{:.1}", t * 1e6))
                };
                let ratio = |a: &Option<(f64, Vec<f32>)>, b: &Option<(f64, Vec<f32>)>| match (a, b)
                {
                    (Some((ta, _)), Some((tb, _))) => format!("{:.2}", tb / ta),
                    _ => "–".to_owned(),
                };
                let names = ["hand bm", "hand cm", "library bm", "library cm"];
                let best = cells
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| c.as_ref().map(|(t, _)| (i, *t)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map_or("–", |(i, _)| names[i]);
                println!(
                    "| {problem} | {p} | {} | {} | {} | {} | {} | {} | {} | {} | {best} |",
                    level.level,
                    level.pairs(),
                    us(&cells[0]),
                    us(&cells[1]),
                    ratio(&cells[0], &cells[1]),
                    us(&cells[2]),
                    us(&cells[3]),
                    ratio(&cells[2], &cells[3]),
                );
            }
        }
    }
    println!();
}

/// The host tables at p ≤ 8 (dense, for the host operator) and the 316 dense f32 M2L
/// matrices of the device (built apart: the host's are not public; with a cache both
/// load).
fn tables(p: usize, cache: Option<&TableCache>) -> (Option<HostTables<f32>>, Vec<f32>) {
    eprintln!("p = {p}: building the tables");
    let host = (p <= 8).then(|| match cache {
        Some(cache) => HostTables::<f32>::load_or_build(p, M2lStrategy::Dense, cache).0,
        None => HostTables::<f32>::build(p, M2lStrategy::Dense),
    });
    let m2l: M2lTables<f32> = match cache {
        Some(cache) => cache.load_or_build(p).0,
        None => M2lTables::build(p),
    };
    (host, m2l.matrices().as_slice().to_vec())
}

fn print_header(device: &Device, arguments: &Arguments) {
    println!("# The device dense M2L per level (Phase 4 T9, C4.5)");
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
         measured); GFLOP/s counts 2 n² per pair (useful flops) |"
    );
    println!(
        "| host | `LaplaceOperator::m2l_pair` with the dense f32 tables, 1 and {} scoped \
         threads |",
        arguments.threads
    );
    println!("| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |");
    println!();
}

/// The settings of a level's plan in f32 at order n.
fn settings(device: &Device, n: usize, policy: GemmPolicy, budget: u64) -> PlanSettings {
    PlanSettings {
        n,
        layout: GemmLayout::default_for(device.info(), n),
        policy,
        budget,
        orientation: Orientation::BoxMajor,
    }
}

/// A level's plan with `settings`, its scratch.
fn plan_of(
    device: &mut Device,
    level: &Level,
    settings: &PlanSettings,
    tables: &Tables<f32>,
) -> (GroupedPlan, TranslationScratch<f32>) {
    let n = settings.n;
    let size = settings.size(device.backend(), Precision::F32, &level.batch_offsets);
    let mut scratch = TranslationScratch::<f32>::new(device, size.columns * n).unwrap();
    let plan = GroupedPlan::new(
        device,
        &level.arrays(),
        level.boxes(),
        settings,
        tables,
        &mut scratch,
    )
    .unwrap();
    (plan, scratch)
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

/// The relative L2 difference of `a` from `b`.
fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut r) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        r += f64::from(y).powi(2);
    }
    if r > 0.0 { (d / r).sqrt() } else { d.sqrt() }
}

/// Times every V level at degree p.
fn measure(
    device: &mut Device,
    (problem, p): (&'static str, usize),
    views: &[Level],
    m2l: &[f32],
    host: Option<LaplaceOperator<f32>>,
    threads: usize,
) -> Vec<Row> {
    let n = (p + 1) * (p + 1);
    let auto = settings(device, n, GemmPolicy::Auto, DEFAULT_SCRATCH_BYTES);
    let library_copy = auto.library_candidate(device.backend(), Precision::F32);
    let tables = Tables::upload(device, m2l, n, library_copy).unwrap();
    let mut rng = SplitMix64::new(0x7e_9001 + p as u64);
    warm_up(device);
    let mut rows = Vec::new();
    for level in views {
        eprintln!("{problem}, p = {p}: M2L on level {}", level.level);
        let boxes = level.boxes();
        let multipoles_host: Vec<f32> = (0..boxes * n)
            .map(|i| rng.range(-1.0, 1.0) as f32 / (1.0 + (i % n) as f32))
            .collect();
        let multipoles = device.upload(&multipoles_host).unwrap();
        let zeros = vec![0.0f32; boxes * n];
        let mut locals = device.upload(&zeros).unwrap();
        let view = GroupedView::upload(device, &level.arrays(), boxes).unwrap();
        let (plan, mut scratch) = plan_of(device, level, &auto, &tables);

        // The level call from zero, for the check.
        let call = |device: &mut Device,
                    plan: &GroupedPlan,
                    scratch: &mut TranslationScratch<f32>,
                    locals: &mut DeviceBuffer<f32>| {
            grouped(
                device,
                plan,
                &view,
                Accumulate::Rows,
                &tables,
                Operands::Separate {
                    input: multipoles.as_slice(),
                    output: locals.as_slice_mut(),
                },
                scratch,
            )
            .unwrap();
        };
        call(device, &plan, &mut scratch, &mut locals);
        let mut got = vec![0.0f32; boxes * n];
        device.download(locals.as_slice(), &mut got).unwrap();

        // The hand-written GEMM on the same batches, where the plan took the library.
        let hand_settings = settings(device, n, GemmPolicy::HandWritten, DEFAULT_SCRATCH_BYTES);
        let mut hand =
            (plan.gemm() == Gemm::Library).then(|| plan_of(device, level, &hand_settings, &tables));

        // The check: the host's level call (p ≤ 8), or the hand-written kernel's.
        let (difference, check, host_times) = match &host {
            Some(op) => {
                let (want, one) = host_level(level, p, &multipoles_host, op, 1);
                let (_, many) = host_level(level, p, &multipoles_host, op, threads);
                (relative_l2(&got, &want), "host", Some((one, many)))
            }
            None => match &mut hand {
                Some((hand_plan, hand_scratch)) => {
                    device.write(locals.as_slice_mut(), &zeros).unwrap();
                    call(device, hand_plan, hand_scratch, &mut locals);
                    let mut want = vec![0.0f32; boxes * n];
                    device.download(locals.as_slice(), &mut want).unwrap();
                    (relative_l2(&got, &want), "hand-written", None)
                }
                None => (0.0, "none", None),
            },
        };

        // The stages and the whole call.
        let stage_time = |device: &mut Device,
                          plan: &GroupedPlan,
                          scratch: &mut TranslationScratch<f32>,
                          locals: &mut DeviceBuffer<f32>,
                          stage: Stage| {
            time(device, |d| {
                grouped_stage(
                    d,
                    plan,
                    &view,
                    Accumulate::Rows,
                    &tables,
                    Operands::Separate {
                        input: multipoles.as_slice(),
                        output: locals.as_slice_mut(),
                    },
                    scratch,
                    stage,
                )
                .unwrap();
            })
        };
        let gather = stage_time(device, &plan, &mut scratch, &mut locals, Stage::Gather);
        let gemm_time = stage_time(device, &plan, &mut scratch, &mut locals, Stage::Gemm);
        let accumulate = stage_time(device, &plan, &mut scratch, &mut locals, Stage::Accumulate);
        let whole = time(device, |d| call(d, &plan, &mut scratch, &mut locals));
        let hand_time = hand.map(|(hand_plan, mut hand_scratch)| {
            stage_time(
                device,
                &hand_plan,
                &mut hand_scratch,
                &mut locals,
                Stage::Gemm,
            )
        });

        // Structure (A) with the plan's GEMM.
        let (offsets, k_mean, k_max) = level.batches();
        let mut scratch_a = TranslationScratch::<f32>::new(device, k_max.max(1) * n).unwrap();
        // The library rejects some narrow per-offset shapes; (A) then runs the hand-written
        // kernel, and the row says so.
        let mut plan_a = |gemm| {
            PerGroupPlan::new(
                device,
                &level.arrays(),
                boxes,
                n,
                gemm,
                &tables,
                &mut scratch_a,
            )
        };
        let plan_a = match plan_a(plan.gemm()) {
            Ok(plan_a) => plan_a,
            Err(error) => {
                eprintln!("  (A) with the library: {error}; (A) runs the hand-written kernel");
                plan_a(Gemm::HandWritten(hand_settings.layout)).unwrap()
            }
        };
        let per_offset = time(device, |d| {
            per_group(
                d,
                &plan_a,
                &view,
                &tables,
                Operands::Separate {
                    input: multipoles.as_slice(),
                    output: locals.as_slice_mut(),
                },
                &mut scratch_a,
            )
            .unwrap();
        });
        rows.push(Row {
            level: level.level,
            boxes,
            pairs: level.pairs(),
            offsets,
            k_mean,
            k_max,
            gemm: plan.gemm(),
            chunks: plan.nchunks(),
            gemm_columns: plan.gemm_columns(),
            rejection: plan.library_rejection().map(str::to_owned),
            gemm_time,
            gather,
            accumulate,
            call: whole,
            hand: hand_time,
            per_offset,
            per_offset_launches: 3 * plan_a.nonempty_groups(),
            per_offset_gemm: plan_a.gemm(),
            host: host_times,
            difference,
            check,
        });
    }
    rows
}

/// The host operator's M2L of `level` from zero locals on `threads` scoped threads, each a
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

/// The spike's cell nearest (p, columns): (p, B, best hand-written, library).
fn spike(p: usize, columns: f64) -> (usize, usize, f64, Option<f64>) {
    let i = (0..SPIKE_DEGREES.len())
        .min_by_key(|&i| SPIKE_DEGREES[i].abs_diff(p))
        .unwrap();
    let ln = columns.max(1.0).ln();
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

fn gemm_name(gemm: Gemm) -> &'static str {
    match gemm {
        Gemm::Library => "library",
        Gemm::HandWritten(_) => "hand-written",
    }
}

fn print_levels(problem: &str, p: usize, rows: &[Row], device: &Device, threads: usize) {
    let n = (p + 1) * (p + 1);
    let flops = |pairs: usize| 2.0 * (n * n * pairs) as f64;
    println!("## {problem}, p = {p} (n = {n}), {}", device.backend());
    println!();
    println!(
        "GEMM per level: the plan's GEMM over every chunk (useful GFLOP/s), the library's \
         padding, and the spike at the nearest (p, B) for the mean columns per offset k."
    );
    println!();
    println!(
        "| level | boxes | pairs | offsets | k mean / max | plan's GEMM (chunks) | useful \
         columns | GEMM µs | GFLOP/s | % peak | hand-written GEMM µs (GFLOP/s) | spike (p, B) \
         GFLOP/s | of the spike |"
    );
    println!(
        "| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |"
    );
    for r in rows {
        let g = flops(r.pairs) / r.gemm_time / 1e9;
        let (sp, sb, shand, slib) = spike(p, r.k_mean);
        let reference = match r.gemm {
            Gemm::Library => slib,
            Gemm::HandWritten(_) => Some(shand),
        };
        println!(
            "| {} | {} | {} | {} | {:.0} / {} | {} ({}) | {:.0}% | {:.1} | {:.0} | {:.1} | {} | \
             ({sp}, {sb}): {} | {} |",
            r.level,
            r.boxes,
            r.pairs,
            r.offsets,
            r.k_mean,
            r.k_max,
            gemm_name(r.gemm),
            r.chunks,
            100.0 * r.pairs as f64 / r.gemm_columns.max(1) as f64,
            r.gemm_time * 1e6,
            g,
            100.0 * g / PEAK_GFLOPS,
            r.hand.map_or("–".to_owned(), |t| format!(
                "{:.1} ({:.0})",
                t * 1e6,
                flops(r.pairs) / t / 1e9
            )),
            reference.map_or("rejected".to_owned(), |v| format!("{v:.0}")),
            reference.map_or("–".to_owned(), |v| format!("{:.0}%", 100.0 * g / v)),
        );
        if let Some(reason) = &r.rejection {
            println!("|  | library rejected: {reason} | | | | | | | | | | | |");
        }
    }
    println!();
    println!(
        "Stages per level: gather, GEMM and accumulation (each over every chunk), the level \
         call (structure (B), 3 launches per chunk), structure (A) (3 launches per offset, \
         with the plan's GEMM unless the library rejects an offset's shape), and the host."
    );
    println!();
    println!(
        "| level | gather µs | GEMM µs | accumulate µs | level call µs | launches | (A) µs | (A) \
         launches | (A) GEMM | host 1 thread µs | host {threads} threads µs | x host 1 | x host \
         {threads} | relative L2 (against) |"
    );
    println!(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: \
         | --- |"
    );
    for r in rows {
        let (h1, hn, x1, xn) = match r.host {
            Some((one, many)) => (
                format!("{:.1}", one * 1e6),
                format!("{:.1}", many * 1e6),
                format!("{:.1}", one / r.call),
                format!("{:.1}", many / r.call),
            ),
            None => ("–".into(), "–".into(), "–".into(), "–".into()),
        };
        println!(
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {:.1} | {} | {} | {h1} | {hn} | {x1} | \
             {xn} | {:.1e} ({}) |",
            r.level,
            r.gather * 1e6,
            r.gemm_time * 1e6,
            r.accumulate * 1e6,
            r.call * 1e6,
            3 * r.chunks,
            r.per_offset * 1e6,
            r.per_offset_launches,
            gemm_name(r.per_offset_gemm),
            r.difference,
            r.check,
        );
    }
    println!();
}

fn print_totals(problem: &str, totals: &[(usize, Vec<Row>)], threads: usize) {
    println!("## {problem}: the M2L stage (every V level)");
    println!();
    println!(
        "| p | levels | pairs | device level calls µs | of which GEMM / gather / accumulate | \
         launches | (A) µs | (A) launches | host 1 thread µs | host {threads} threads µs | x \
         host 1 | x host {threads} |"
    );
    println!(
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for (p, rows) in totals {
        let sum = |f: &dyn Fn(&Row) -> f64| rows.iter().map(f).sum::<f64>();
        let call = sum(&|r| r.call);
        let host = rows
            .iter()
            .map(|r| r.host)
            .collect::<Option<Vec<_>>>()
            .map(|h| {
                (
                    h.iter().map(|x| x.0).sum::<f64>(),
                    h.iter().map(|x| x.1).sum::<f64>(),
                )
            });
        println!(
            "| {p} | {} | {} | {:.1} | {:.0}% / {:.0}% / {:.0}% | {} | {:.1} | {} | {} | {} | {} | \
             {} |",
            rows.len(),
            rows.iter().map(|r| r.pairs).sum::<usize>(),
            call * 1e6,
            100.0 * sum(&|r| r.gemm_time) / call,
            100.0 * sum(&|r| r.gather) / call,
            100.0 * sum(&|r| r.accumulate) / call,
            rows.iter().map(|r| 3 * r.chunks).sum::<usize>(),
            sum(&|r| r.per_offset) * 1e6,
            rows.iter().map(|r| r.per_offset_launches).sum::<usize>(),
            host.map_or("–".into(), |h| format!("{:.1}", h.0 * 1e6)),
            host.map_or("–".into(), |h| format!("{:.1}", h.1 * 1e6)),
            host.map_or("–".into(), |h| format!("{:.1}", h.0 / call)),
            host.map_or("–".into(), |h| format!("{:.1}", h.1 / call)),
        );
    }
    println!();
}

/// The C4.5 gate: the GEMM alone on the spike's shapes (module documentation).
fn gate(device: &mut Device) {
    warm_up(device);
    println!("## The gate: the GEMM alone on the spike's shapes (one table, B columns)");
    println!();
    println!(
        "| p | B | GEMM of the rule | µs | GFLOP/s | % peak | spike GFLOP/s | of the spike | \
         gate (>= 80%) | hand-written µs | GFLOP/s | spike hand-written | of it |"
    );
    println!(
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"
    );
    let mut rng = SplitMix64::new(0x7e_9100);
    for (i, &p) in SPIKE_DEGREES.iter().enumerate() {
        let n = (p + 1) * (p + 1);
        let table: Vec<f32> = (0..n * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
        let tables = Tables::upload(device, &table, n, false).unwrap();
        let layout = GemmLayout::default_for(device.info(), n);
        for (j, &b) in SPIKE_COLUMNS.iter().enumerate() {
            eprintln!("gate: p = {p}, B = {b}");
            let x_host: Vec<f32> = (0..b * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
            let x = device.upload(&x_host).unwrap();
            let mut y = device.alloc::<f32>(b * n).unwrap();
            let schedule = TileSchedule::new(device, layout, &[0, b as u32]).unwrap();
            let hand = time(device, |d| {
                gemm(
                    d,
                    layout,
                    n,
                    tables.compact(),
                    &schedule,
                    x.as_slice(),
                    y.as_slice_mut(),
                )
                .unwrap();
            });
            let rule_library = p >= 8 && device.backend().is_gpu();
            let library_time = if rule_library {
                let probe = library(
                    device,
                    (1, b, n),
                    tables.compact(),
                    x.as_slice(),
                    y.as_slice_mut(),
                );
                match probe {
                    Ok(()) => Some(time(device, |d| {
                        library(
                            d,
                            (1, b, n),
                            tables.compact(),
                            x.as_slice(),
                            y.as_slice_mut(),
                        )
                        .unwrap();
                    })),
                    Err(error) => {
                        eprintln!("gate: p = {p}, B = {b}: the library rejects it: {error}");
                        None
                    }
                }
            } else {
                None
            };
            let flops = 2.0 * (n * n * b) as f64;
            let g = |t: f64| flops / t / 1e9;
            let (name, t, reference) = match library_time {
                Some(t) => ("library", t, SPIKE_LIBRARY[i][j]),
                None => ("hand-written", hand, Some(SPIKE_HAND[i][j])),
            };
            let ratio = reference.map(|r| g(t) / r);
            println!(
                "| {p} | {b} | {name} | {:.1} | {:.0} | {:.1} | {} | {} | {} | {:.1} | {:.0} | {:.0} \
                 | {:.0}% |",
                t * 1e6,
                g(t),
                100.0 * g(t) / PEAK_GFLOPS,
                reference.map_or("–".into(), |r| format!("{r:.0}")),
                ratio.map_or("–".into(), |r| format!("{:.0}%", 100.0 * r)),
                ratio.map_or("–".into(), |r| if r >= GATE { "met" } else { "not met" }
                    .to_owned()),
                hand * 1e6,
                g(hand),
                SPIKE_HAND[i][j],
                100.0 * g(hand) / SPIKE_HAND[i][j],
            );
        }
    }
    println!();
}

/// The level call of `level` at p = 8 under several scratch budgets (module
/// documentation).
fn budgets(device: &mut Device, level: &Level, m2l: &[f32]) {
    let p = 8;
    let n = (p + 1) * (p + 1);
    warm_up(device);
    println!(
        "## The scratch budget: the cube's level {} at p = 8 ({} pairs)",
        level.level,
        level.pairs()
    );
    println!();
    println!("| budget MB | GEMM | chunks | scratch MB | level call µs | launches |");
    println!("| ---: | --- | ---: | ---: | ---: | ---: |");
    let auto = settings(device, n, GemmPolicy::Auto, DEFAULT_SCRATCH_BYTES);
    let library_copy = auto.library_candidate(device.backend(), Precision::F32);
    let tables = Tables::upload(device, m2l, n, library_copy).unwrap();
    let boxes = level.boxes();
    let mut rng = SplitMix64::new(0x7e_9200);
    let multipoles_host: Vec<f32> = (0..boxes * n)
        .map(|_| rng.range(-1.0, 1.0) as f32)
        .collect();
    let multipoles = device.upload(&multipoles_host).unwrap();
    let mut locals = device.alloc::<f32>(boxes * n).unwrap();
    let view = GroupedView::upload(device, &level.arrays(), boxes).unwrap();
    for mb in [8u64, 16, 32, 64, 128, 256, 512] {
        eprintln!("budget: {mb} MB");
        let s = settings(device, n, GemmPolicy::Auto, mb << 20);
        let (plan, mut scratch) = plan_of(device, level, &s, &tables);
        let seconds = time(device, |d| {
            grouped(
                d,
                &plan,
                &view,
                Accumulate::Rows,
                &tables,
                Operands::Separate {
                    input: multipoles.as_slice(),
                    output: locals.as_slice_mut(),
                },
                &mut scratch,
            )
            .unwrap();
        });
        println!(
            "| {mb} | {} | {} | {:.1} | {:.1} | {} |",
            gemm_name(plan.gemm()),
            plan.nchunks(),
            TranslationScratch::<f32>::bytes(plan.columns() * n) as f64 / f64::from(1 << 20),
            seconds * 1e6,
            3 * plan.nchunks()
        );
    }
    println!();
}

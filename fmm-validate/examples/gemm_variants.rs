//! The variants of the device's grouped translations (Phase 6 T7, C6.1): every M2L, M2M
//! and L2L level call of an FMM's plan with the hand-written GEMM, timed under each
//! variant of `nd_fmm_kernels::translate` that T7 adds, beside Phase 4S's code, and the
//! results compared bit for bit. Prints Markdown on stdout and progress on stderr. Feature
//! `gpu` (with a backend).
//!
//! The variants (`PlanSettings`, the hand-written GEMM in every row):
//!
//! | name | layout | gather | reduction |
//! | --- | --- | --- | --- |
//! | `phase4s` | the backend's default (`GemmLayout::default_for`) | separate | walking every row (`Reduction::Walk`) |
//! | `by-chunk` | default | separate | by chunk (`Reduction::ByChunk`) |
//! | `fused` | default | fused (`Gather::Fused`) | by chunk |
//! | `shared` | `GemmLayout::shared_for` | separate | by chunk |
//! | `shared-fused` | `GemmLayout::shared_for` | fused | by chunk |
//!
//! For the problem (`--problem cube|plummer`, N = `--n`, default 10⁶, the C3.3 tree
//! settings: `max_level` 16, 64 points per leaf) the host `Fmm` is built once for its plan;
//! for each p of `--degrees` (default 3, 6, 8) and each level call of the kinds of
//! `--kinds` (default m2l,m2m,l2l) with a pair, with seeded random inputs: each variant's
//! plan is built with the chunk budget of `--budget MB` (default the device's,
//! `default_scratch_bytes`: 128 MB on Metal, 2 GB on CUDA), its level call run once from
//! zero outputs and compared with `phase4s`'s bit for bit (the count of differing values is
//! printed; every variant keeps the order, so it must be 0), then timed whole and, for
//! `phase4s` and `fused`, stage by stage (`grouped_stage`: the gather, the GEMM and the
//! accumulation over every chunk). With `--vectors m` each view has m columns per box, as
//! `nd-fmm-exec` builds them for m charge vectors (Phase 6 T5, `device::Columns`): row
//! t m + v holds row t's entries with sources s m + v, each batch's pairs become m pairs.
//!
//! Timing: half a second of other device work before each table (the clocks ramp down while
//! the host builds); launches queued between syncs, after a warm-up launch and a sync
//! (compilation excluded); each figure the median of 15 batches of at least 20 ms
//! (`nd_fmm_validate::bench::median_time_per_call`). GFLOP/s counts 2 n² flops per pair,
//! n = (p + 1)²; % of peak against `nd_fmm_validate::peaks` in the run's precision. Nothing
//! is asserted but the bits.
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --features metal --example gemm_variants -- \
//!     --device metal [--precision f32] [--n 1e6] [--degrees 3,6,8] [--table-cache DIR]
//! ```
//!
//! Metal needs a process with GPU access (outside the macOS sandbox); on locust
//! `--features cuda --device cuda`. It initialises MPI and runs on one rank.

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_kernels::translate::{
    Accumulate, Gather, GemmLayout, GemmPolicy, GroupedPlan, Operands, PlanSettings, Reduction,
    Stage, Tables, TranslationScratch, default_scratch_bytes, grouped, grouped_stage,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_tables::{L2lTables, M2lTables, M2mTables, TableCache};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target, toolchain};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};
use nd_fmm_validate::peaks::{self, Peaks};

/// The float types the example times, f32 and f64.
trait Value: DeviceFloat + Stored {}

impl<T: DeviceFloat + Stored> Value for T {}

/// `x` in the precision of T, rounded to nearest.
fn value<T: Value>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// The kinds of level call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    M2l,
    M2m,
    L2l,
}

impl Kind {
    fn accumulate(self) -> Accumulate {
        match self {
            Self::L2l => Accumulate::Scatter,
            _ => Accumulate::Rows,
        }
    }
}

/// A variant of the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Phase4s,
    ByChunk,
    Fused,
    Shared,
    SharedFused,
}

impl Variant {
    const ALL: [Self; 5] = [
        Self::Phase4s,
        Self::ByChunk,
        Self::Fused,
        Self::Shared,
        Self::SharedFused,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Phase4s => "phase4s",
            Self::ByChunk => "by-chunk",
            Self::Fused => "fused",
            Self::Shared => "shared",
            Self::SharedFused => "shared-fused",
        }
    }

    /// The plan settings of the variant at order n with `budget`, or `None` where the
    /// device has no shared-memory layout of order n.
    fn settings(
        self,
        device: &Device,
        (n, precision): (usize, Precision),
        budget: u64,
    ) -> Option<PlanSettings> {
        let default = GemmLayout::default_for(device.info(), n);
        let shared = GemmLayout::shared_for(device.info(), n, precision);
        let (layout, gather, reduction) = match self {
            Self::Phase4s => (default, Gather::Separate, Reduction::Walk),
            Self::ByChunk => (default, Gather::Separate, Reduction::ByChunk),
            Self::Fused => (default, Gather::Fused, Reduction::ByChunk),
            Self::Shared => (shared?, Gather::Separate, Reduction::ByChunk),
            Self::SharedFused => (shared?, Gather::Fused, Reduction::ByChunk),
        };
        Some(PlanSettings {
            gather,
            reduction,
            ..PlanSettings::new(n, layout, GemmPolicy::HandWritten, budget)
        })
    }
}

/// The command line.
struct Arguments {
    device: String,
    precision: Precision,
    n: usize,
    problem: Distribution,
    degrees: Vec<usize>,
    kinds: Vec<Kind>,
    budget: Option<u64>,
    vectors: usize,
    table_cache: Option<String>,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: gemm_variants --device cpu|metal|cuda [--precision f32|f64] [--n N] \
             [--problem cube|plummer] [--degrees p,p,...] [--kinds m2l,m2m,l2l] \
             [--budget MB] [--vectors m] [--table-cache DIR]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        precision: Precision::F32,
        n: 1_000_000,
        problem: Distribution::Cube,
        degrees: vec![3, 6, 8],
        kinds: vec![Kind::M2l, Kind::M2m, Kind::L2l],
        budget: None,
        vectors: 1,
        table_cache: None,
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--device", [value, tail @ ..]) => {
                parsed.device = value.clone();
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
                parsed.n = n as usize;
                tail
            }
            ("--problem", [value, tail @ ..]) => {
                parsed.problem = match value.as_str() {
                    "cube" => Distribution::Cube,
                    "plummer" => Distribution::Plummer,
                    _ => usage(),
                };
                tail
            }
            ("--degrees", [value, tail @ ..]) => {
                parsed.degrees = value
                    .split(',')
                    .map(|p| p.parse().unwrap_or_else(|_| usage()))
                    .collect();
                tail
            }
            ("--kinds", [value, tail @ ..]) => {
                parsed.kinds = value
                    .split(',')
                    .map(|k| match k {
                        "m2l" => Kind::M2l,
                        "m2m" => Kind::M2m,
                        "l2l" => Kind::L2l,
                        _ => usage(),
                    })
                    .collect();
                tail
            }
            ("--budget", [value, tail @ ..]) => {
                let mb: u64 = value.parse().unwrap_or_else(|_| usage());
                parsed.budget = Some(mb << 20);
                tail
            }
            ("--vectors", [value, tail @ ..]) => {
                parsed.vectors = value.parse().unwrap_or_else(|_| usage());
                tail
            }
            ("--table-cache", [value, tail @ ..]) => {
                parsed.table_cache = Some(value.clone());
                tail
            }
            _ => usage(),
        };
    }
    if parsed.device.is_empty()
        || parsed.n == 0
        || parsed.vectors == 0
        || parsed.degrees.is_empty()
        || parsed.kinds.is_empty()
    {
        usage();
    }
    parsed
}

/// A grouped view of the plan, copied, with m columns per box: its arrays, the columns of
/// its input and output, and its level.
struct View {
    kind: Kind,
    name: &'static str,
    level: usize,
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u16>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
    inputs: usize,
    outputs: usize,
}

/// The arrays of a grouped view of the plan, the groups widened: its row offsets,
/// sources, groups, batch offsets, batch targets and batch sources.
type Csr<'a> = (
    &'a [u32],
    &'a [u32],
    Vec<u16>,
    &'a [u32],
    &'a [u32],
    &'a [u32],
);

/// The [`Csr`] of a grouped view of the plan.
macro_rules! csr {
    ($view:expr) => {{
        let view = $view;
        (
            view.row_offsets(),
            view.sources(),
            view.groups().iter().map(|&g| u16::from(g)).collect(),
            view.batch_offsets(),
            view.batch_targets(),
            view.batch_sources(),
        )
    }};
}

impl View {
    /// The view of `csr` with m columns per box (module documentation): `inputs` and
    /// `outputs` boxes.
    fn new(
        (kind, name, level): (Kind, &'static str, usize),
        csr: Csr<'_>,
        (inputs, outputs): (usize, usize),
        m: usize,
    ) -> Self {
        let (row_offsets, sources, groups, batch_offsets, batch_targets, batch_sources) = csr;
        let m32 = m as u32;
        let nrows = row_offsets.len() - 1;
        let mut view = Self {
            kind,
            name,
            level,
            row_offsets: vec![0],
            sources: Vec::new(),
            groups: Vec::new(),
            batch_offsets: batch_offsets.iter().map(|&o| o * m32).collect(),
            batch_targets: Vec::new(),
            batch_sources: Vec::new(),
            inputs: inputs * m,
            outputs: outputs * m,
        };
        for t in 0..nrows {
            let row = row_offsets[t] as usize..row_offsets[t + 1] as usize;
            for v in 0..m32 {
                view.sources
                    .extend(sources[row.clone()].iter().map(|&s| s * m32 + v));
                view.groups.extend_from_slice(&groups[row.clone()]);
                view.row_offsets.push(view.sources.len() as u32);
            }
        }
        for (&t, &s) in batch_targets.iter().zip(batch_sources) {
            for v in 0..m32 {
                view.batch_targets.push(t * m32 + v);
                view.batch_sources.push(s * m32 + v);
            }
        }
        view
    }

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

    fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    fn pairs(&self) -> usize {
        self.sources.len()
    }
}

/// The views of the kinds of `kinds` with a pair, every level, m columns per box.
fn views(fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>, kinds: &[Kind], m: usize) -> Vec<View> {
    let plan = fmm.plan();
    let index = plan.index();
    let boxes = |l: usize| index.len(l);
    let mut out = Vec::new();
    for (level, lists) in plan.levels().iter().enumerate() {
        if kinds.contains(&Kind::M2l) && !lists.v().is_empty() {
            out.push(View::new(
                (Kind::M2l, "M2L", level),
                csr!(lists.v()),
                (boxes(level), boxes(level)),
                m,
            ));
        }
        if kinds.contains(&Kind::M2m) {
            for (name, csr) in [
                ("M2M local", lists.m2m_local()),
                ("M2M global", lists.m2m_global()),
            ] {
                if !csr.is_empty() {
                    out.push(View::new(
                        (Kind::M2m, name, level),
                        csr!(csr),
                        (boxes(level + 1), boxes(level)),
                        m,
                    ));
                }
            }
        }
        if kinds.contains(&Kind::L2l) && !lists.l2l().is_empty() {
            out.push(View::new(
                (Kind::L2l, "L2L", level),
                csr!(lists.l2l()),
                (boxes(level - 1), boxes(level)),
                m,
            ));
        }
    }
    out
}

/// One variant of one level call.
struct Timed {
    variant: Variant,
    chunks: usize,
    row_pairs: usize,
    call: f64,
    stages: Option<[f64; 3]>,
    differing: usize,
}

/// One level call under every variant.
struct Row {
    name: &'static str,
    level: usize,
    rows: usize,
    pairs: usize,
    timed: Vec<Timed>,
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
            "--precision {:?}: {} does no arithmetic in it",
            arguments.precision,
            device.info()
        );
        std::process::exit(1);
    }
    match arguments.precision {
        Precision::F32 => run::<f32>(&mut device, &arguments),
        Precision::F64 => run::<f64>(&mut device, &arguments),
    }
}

/// The whole run in the precision of T.
fn run<T: Value>(device: &mut Device, arguments: &Arguments) {
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises once");
    let comm = universe.world();
    let cache = arguments.table_cache.as_deref().map(TableCache::new);
    let config = Config {
        n: arguments.n,
        ..Config::c33(arguments.problem)
    };
    let problem_name = match arguments.problem {
        Distribution::Plummer => "Plummer sphere",
        _ => "uniform cube",
    };
    let budget = arguments
        .budget
        .unwrap_or_else(|| default_scratch_bytes(device.info()));
    print_header::<T>(device, arguments, problem_name, budget);
    let problem = Problem::new(&config);
    eprintln!("{problem_name}: building the host FMM for its plan");
    let fmm = FmmBuilder::<f32>::new(3)
        .strategy(M2lStrategy::Rotation)
        .max_level(config.max_level)
        .max_points_per_leaf(config.max_points_per_leaf)
        .build(&problem.points, &problem.points, &comm)
        .expect("the host FMM builds on one rank");
    let views = views(&fmm, &arguments.kinds, arguments.vectors);
    drop(fmm);
    for &p in &arguments.degrees {
        let rows = measure::<T>(device, p, &views, budget, cache.as_ref());
        print_rows::<T>(device, p, &rows);
    }
}

/// The tables of `kind` at degree p, n² values each.
fn tables<T: Value>(kind: Kind, p: usize, cache: Option<&TableCache>) -> Vec<T> {
    match kind {
        Kind::M2l => {
            let tables: M2lTables<T> = match cache {
                Some(cache) => cache.load_or_build(p).0,
                None => M2lTables::build(p),
            };
            tables.matrices().as_slice().to_vec()
        }
        Kind::M2m => M2mTables::<T>::build(p).matrices().as_slice().to_vec(),
        Kind::L2l => L2lTables::<T>::build(p).matrices().as_slice().to_vec(),
    }
}

/// Times every view at degree p under every variant.
fn measure<T: Value>(
    device: &mut Device,
    p: usize,
    views: &[View],
    budget: u64,
    cache: Option<&TableCache>,
) -> Vec<Row> {
    let n = (p + 1) * (p + 1);
    eprintln!("p = {p}: building the tables");
    let mut uploaded: Vec<(Kind, Tables<T>)> = Vec::new();
    for kind in [Kind::M2l, Kind::M2m, Kind::L2l] {
        if views.iter().any(|v| v.kind == kind) {
            let matrices = tables::<T>(kind, p, cache);
            uploaded.push((kind, Tables::upload(device, &matrices, n, false).unwrap()));
        }
    }
    let mut rng = SplitMix64::new(0x7e_c001 + p as u64);
    warm_up(device);
    let mut rows = Vec::new();
    for view in views {
        eprintln!("p = {p}: {} on level {}", view.name, view.level);
        let tables = &uploaded
            .iter()
            .find(|(k, _)| *k == view.kind)
            .expect("uploaded above")
            .1;
        let input_host: Vec<T> = (0..view.inputs * n)
            .map(|i| value::<T>(rng.range(-1.0, 1.0)) / value::<T>(1.0 + (i % n) as f64))
            .collect();
        let input = device.upload(&input_host).unwrap();
        let zeros = vec![value::<T>(0.0); view.outputs * n];
        let mut output = device.upload(&zeros).unwrap();
        let device_view = GroupedView::upload(device, &view.arrays(), view.inputs).unwrap();
        let call = |device: &mut Device,
                    plan: &GroupedPlan,
                    scratch: &mut TranslationScratch<T>,
                    output: &mut DeviceBuffer<T>,
                    stage: Option<Stage>| {
            let operands = Operands::Separate {
                input: input.as_slice(),
                output: output.as_slice_mut(),
            };
            match stage {
                None => grouped(
                    device,
                    plan,
                    &device_view,
                    view.kind.accumulate(),
                    tables,
                    operands,
                    scratch,
                ),
                Some(stage) => grouped_stage(
                    device,
                    plan,
                    &device_view,
                    view.kind.accumulate(),
                    tables,
                    operands,
                    scratch,
                    stage,
                ),
            }
            .unwrap();
        };
        let mut reference: Option<Vec<T>> = None;
        let mut timed = Vec::new();
        for variant in Variant::ALL {
            let Some(settings) = variant.settings(device, (n, T::FLOAT), budget) else {
                continue;
            };
            let size = settings.size(
                device.backend(),
                T::FLOAT,
                &view.batch_offsets,
                view.nrows(),
            );
            // X only where the plan gathers: none with the fused gather, whose chunks are
            // twice as wide (Phase 6 T7).
            let mut scratch =
                TranslationScratch::<T>::split(device, size.gathered * n, size.columns * n)
                    .unwrap();
            let plan = GroupedPlan::new(
                device,
                &view.arrays(),
                view.inputs,
                &settings,
                tables,
                &mut scratch,
            )
            .unwrap_or_else(|error| panic!("{}: {error}", variant.name()));
            // From zero outputs, against the first variant's bits.
            device.write(output.as_slice_mut(), &zeros).unwrap();
            call(device, &plan, &mut scratch, &mut output, None);
            let mut got = vec![value::<T>(0.0); view.outputs * n];
            device.download(output.as_slice(), &mut got).unwrap();
            let differing = match &reference {
                None => {
                    reference = Some(got);
                    0
                }
                Some(want) => got
                    .iter()
                    .zip(want)
                    .filter(|(a, b)| {
                        RealScalar::to_f64(**a).to_bits() != RealScalar::to_f64(**b).to_bits()
                    })
                    .count(),
            };
            let call_time = time(device, |d| call(d, &plan, &mut scratch, &mut output, None));
            let stages = matches!(variant, Variant::Phase4s | Variant::Fused).then(|| {
                [Stage::Gather, Stage::Gemm, Stage::Accumulate].map(|stage| {
                    time(device, |d| {
                        call(d, &plan, &mut scratch, &mut output, Some(stage))
                    })
                })
            });
            timed.push(Timed {
                variant,
                chunks: plan.nchunks(),
                row_pairs: plan.chunk_row_pairs(),
                call: call_time,
                stages,
                differing,
            });
            drop(plan);
            drop(scratch);
        }
        rows.push(Row {
            name: view.name,
            level: view.level,
            rows: view.nrows(),
            pairs: view.pairs(),
            timed,
        });
    }
    rows
}

/// Runs `run` once and syncs (compilation excluded), then times it: the median per call.
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

/// Half a second of other device work.
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

/// The peak of the device in the precision of T, GFLOP/s, if known.
fn peak<T: Value>(device: &Device) -> Option<f64> {
    let precision = match T::FLOAT {
        Precision::F32 => peaks::Precision::F32,
        Precision::F64 => peaks::Precision::F64,
    };
    Peaks::of_info(device.info()).and_then(|p| p.gflops(precision))
}

fn print_header<T: Value>(device: &Device, arguments: &Arguments, problem: &str, budget: u64) {
    println!("# The device GEMM variants per level call (Phase 6 T7, C6.1)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {} |", device.info());
    println!("| precision | {:?} |", T::FLOAT);
    println!(
        "| problem | {problem}, N = {}, `max_level` 16, 64 points per leaf |",
        arguments.n
    );
    println!(
        "| charge vectors (columns per box) | {} |",
        arguments.vectors
    );
    println!("| chunk budget | {} MB |", budget >> 20);
    let peak = peak::<T>(device).map_or("unknown".into(), |g| format!("{g} GFLOP/s"));
    println!("| peak | {peak}; GFLOP/s counts 2 n² per pair |");
    println!(
        "| timing | median of 15 batches of >= 20 ms; launches queued between syncs, a \
         warm-up launch and a sync first (compilation excluded) |"
    );
    for variant in Variant::ALL {
        let describe = variant
            .settings(device, (81, T::FLOAT), budget)
            .map_or("no layout fits".into(), |s| {
                format!("{}, {}, {} (at p = 8)", s.layout, s.gather, s.reduction)
            });
        println!("| `{}` | {describe} |", variant.name());
    }
    println!();
}

fn print_rows<T: Value>(device: &Device, p: usize, rows: &[Row]) {
    let n = (p + 1) * (p + 1);
    let peak = peak::<T>(device);
    let flops = |pairs: usize| 2.0 * (n * n * pairs) as f64;
    println!("## p = {p} (n = {n})");
    println!();
    print!("| call | level | rows | pairs | chunks | row pairs (by chunk) |");
    for variant in Variant::ALL {
        print!(" {} µs |", variant.name());
    }
    println!(" best / phase4s | best GFLOP/s (% of peak) | bits |");
    print!("| --- | ---: | ---: | ---: | ---: | ---: |");
    for _ in Variant::ALL {
        print!(" ---: |");
    }
    println!(" ---: | ---: | --- |");
    let mut totals = [0.0f64; 5];
    let mut stage_totals = [[0.0f64; 3]; 2];
    for row in rows {
        let base = row.timed[0].call;
        let best = row
            .timed
            .iter()
            .min_by(|a, b| a.call.total_cmp(&b.call))
            .expect("a variant");
        let by_chunk = row.timed.iter().find(|t| t.variant == Variant::ByChunk);
        print!(
            "| {} | {} | {} | {} | {} | {} |",
            row.name,
            row.level,
            row.rows,
            row.pairs,
            row.timed[0].chunks,
            by_chunk.map_or(0, |t| t.row_pairs)
        );
        for variant in Variant::ALL {
            match row.timed.iter().find(|t| t.variant == variant) {
                Some(t) => {
                    print!(" {:.1} |", t.call * 1e6);
                    totals[variant as usize] += t.call;
                }
                None => print!(" – |"),
            }
        }
        let g = flops(row.pairs) / best.call / 1e9;
        let share = peak.map_or(String::new(), |peak| format!(" ({:.1}%)", 100.0 * g / peak));
        let differing: usize = row.timed.iter().map(|t| t.differing).sum();
        println!(
            " {:.2} ({}) | {g:.0}{share} | {} |",
            best.call / base,
            best.variant.name(),
            if differing == 0 {
                "same".to_owned()
            } else {
                format!("{differing} values differ")
            }
        );
        for (i, variant) in [Variant::Phase4s, Variant::Fused].into_iter().enumerate() {
            if let Some(stages) = row
                .timed
                .iter()
                .find(|t| t.variant == variant)
                .and_then(|t| t.stages)
            {
                for (s, &t) in stage_totals[i].iter_mut().zip(&stages) {
                    *s += t;
                }
            }
        }
    }
    print!(
        "| **all** | | | {} | | |",
        rows.iter().map(|r| r.pairs).sum::<usize>()
    );
    for (variant, total) in Variant::ALL.iter().zip(totals) {
        let _ = variant;
        print!(" **{:.1}** |", total * 1e6);
    }
    let best = totals
        .iter()
        .copied()
        .filter(|&t| t > 0.0)
        .fold(f64::INFINITY, f64::min);
    println!(" {:.2} | | |", best / totals[0]);
    println!();
    println!(
        "Stages summed over the calls (µs; `grouped_stage`, one launch per chunk): \
         `phase4s` gather {:.1}, GEMM {:.1}, accumulation {:.1}; `fused` gather {:.1}, GEMM \
         {:.1}, accumulation {:.1}.",
        stage_totals[0][0] * 1e6,
        stage_totals[0][1] * 1e6,
        stage_totals[0][2] * 1e6,
        stage_totals[1][0] * 1e6,
        stage_totals[1][1] * 1e6,
        stage_totals[1][2] * 1e6
    );
    println!();
}

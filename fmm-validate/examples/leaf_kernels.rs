//! The device leaf operators P2M, L2P, P2L and M2P of `nd-fmm-kernels` (Phase 4 T7,
//! C4.3), timed per level of an FMM's plan against the host operator on the same level.
//! Prints Markdown on stdout and progress on stderr. Feature `gpu` (with a backend).
//!
//! For each problem (the C3.2 cube: N = 10⁵, a uniform level-4 tree; the Plummer sphere
//! of C3.3: N = 10⁵, `max_level` 16, 64 points per leaf), each p ∈ {3, 8}, in f32 with
//! gradients:
//!
//! - the host `Fmm` is built once, for its plan and its point counts per leaf; the leaf
//!   stores are filled with seeded random leaf-scaled points and charges, and the
//!   multipoles and locals with seeded random coefficients (the timing does not depend on
//!   the values; every list's geometry is the plan's);
//! - per level and operator with a non-empty view: the device kernel, one launch per call
//!   as the device operator issues it (`nd_fmm_kernels::leaf`), in the backend's default
//!   layout and, on a GPU, the cube layout with 32 units and tiles of 16 points; launches
//!   queued between syncs, a warm-up launch first (compilation excluded), the median of
//!   15 batches of at least 20 ms;
//! - against the host operator on the same level (`LaplaceOperator`'s `p2m_leaf`,
//!   `p2l_pair`, `l2p_leaf` and `m2p_pair`, the bodies of its level calls) on one thread
//!   and on `--threads` scoped threads (default 12, the performance cores), each a
//!   contiguous share of the level's rows, with every BLAS thread variable set to 1 by the
//!   launcher;
//! - first, every device result is compared with the host's on the same inputs (relative
//!   L2 over the level's outputs, from zero), so that a timed row is a correct one.
//!
//! The output is a table per problem and p: device time per launch, host times, and the
//! speed-ups; then the totals per operator. Nothing is asserted. f64 is not timed on a GPU
//! (Metal has none) and CUDA is type-checked only.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --features metal --example leaf_kernels -- \
//!     --device metal
//! ```
//!
//! `--device cpu` runs the same on the CubeCL CPU runtime (correctness backend; its
//! timings are not a Phase 4 target). `--quick` runs the C3.2 cube at p = 3 only. Metal
//! needs a process with GPU access (outside the macOS sandbox). It initialises MPI and
//! runs on one rank.

use std::time::Instant;

use mpi::Threading;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_kernels::leaf::{LeafLayout, SourceInputs, TargetInputs, l2p, m2p, p2l, p2m};
use nd_fmm_kernels::view::{BoxCoordinates, IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{
    cores, cpu_model, median_time_per_call, performance_cores, target, toolchain,
};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution, Problem};

/// The command line.
struct Arguments {
    device: String,
    threads: usize,
    quick: bool,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: leaf_kernels --device cpu|metal|cuda [--threads n] [--quick]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        device: String::new(),
        threads: 12,
        quick: false,
    };
    let mut rest = args.as_slice();
    while let [flag, tail @ ..] = rest {
        rest = match (flag.as_str(), tail) {
            ("--device", [value, tail @ ..]) => {
                parsed.device = value.clone();
                tail
            }
            ("--threads", [value, tail @ ..]) => {
                parsed.threads = value
                    .parse()
                    .ok()
                    .filter(|&n| n >= 1)
                    .unwrap_or_else(|| usage());
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

/// The four operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    P2m,
    P2l,
    L2p,
    M2p,
}

impl Kind {
    const ALL: [Self; 4] = [Self::P2m, Self::P2l, Self::L2p, Self::M2p];

    fn name(self) -> &'static str {
        match self {
            Self::P2m => "P2M",
            Self::P2l => "P2L",
            Self::L2p => "L2P",
            Self::M2p => "M2P",
        }
    }
}

/// One timed (problem, p, operator, level).
struct Row {
    problem: &'static str,
    p: usize,
    kind: Kind,
    level: usize,
    /// Rows of the view and entries.
    rows: usize,
    entries: usize,
    /// Points the level's call touches: sources (P2M, P2L) or targets (L2P, M2P).
    points: usize,
    /// Device seconds per launch, by layout.
    device: Vec<(LeafLayout, f64, f64)>,
    host_one: f64,
    host_many: f64,
}

/// The data of one problem's plan on the host and on the device.
struct Data {
    nlevels: usize,
    /// Per level: the views (rows, entries) of P2M, P2L, L2P and M2P.
    views: Vec<[(Vec<u32>, Vec<u32>); 4]>,
    /// Per level: the Morton keys of the boxes, and the local leaves of the level.
    box_keys: Vec<Vec<u64>>,
    /// The Morton key and level of every leaf.
    leaf_keys: Vec<u64>,
    leaf_levels: Vec<usize>,
    local: Vec<std::ops::Range<usize>>,
    /// The source and target stores (CONVENTIONS §3.13) and their point offsets.
    sources: Vec<f32>,
    source_offsets: Vec<usize>,
    targets: Vec<f32>,
    target_offsets: Vec<usize>,
    /// Random coefficients of every level, box after box.
    coefficients: Vec<Vec<f32>>,
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
    print_header(&device, &arguments);
    let mut problems = vec![("C3.2 cube", Config::C32)];
    if !arguments.quick {
        problems.push(("Plummer sphere", Config::c33(Distribution::Plummer)));
    }
    let degrees: &[usize] = if arguments.quick { &[3] } else { &[3, 8] };
    for (name, config) in problems {
        let problem = Problem::new(&config);
        for &p in degrees {
            eprintln!("{name}, p = {p}: building the host FMM");
            let fmm = FmmBuilder::<f32>::new(p)
                .gradients(true)
                .max_level(config.max_level)
                .max_points_per_leaf(config.max_points_per_leaf)
                .build(&problem.points, &problem.points, &comm)
                .expect("the host FMM builds on one rank");
            let data = data(&fmm, p);
            let rows = measure(
                &mut device,
                (name, p),
                &data,
                fmm.operator(),
                arguments.threads,
            );
            print_table(&rows, &device, arguments.threads);
        }
    }
}

fn print_header(device: &Device, arguments: &Arguments) {
    println!("# The device leaf operators per level (Phase 4 T7, C4.3)");
    println!();
    println!("| item | value |");
    println!("| --- | --- |");
    println!("| machine | {} ({}) |", cpu_model(), cores());
    println!(
        "| performance cores | {} |",
        performance_cores().map_or("unknown".into(), |p| p.to_string())
    );
    println!("| target | {} |", target());
    println!("| toolchain | {} |", toolchain());
    println!("| device | {} |", device.info());
    println!(
        "| host threads of the many-thread column | {} |",
        arguments.threads
    );
    println!(
        "| timing | median of 15 batches of >= 20 ms; device launches queued between syncs, a \
         warm-up launch first (compilation excluded); host: the host operator's per-target \
         bodies on 1 thread and on scoped threads, each a contiguous share of the rows |"
    );
    println!("| precision | f32 (f64 not timed: Metal has no f64; CUDA type-checked only) |");
    println!();
}

/// The plan's views and keys, random stores and coefficients.
fn data(fmm: &nd_fmm_exec::fmm::Fmm<'_, f32>, p: usize) -> Data {
    let plan = fmm.plan();
    let index = plan.index();
    let nlevels = plan.nlevels();
    let csr = |rows: &[u32], entries: &[u32]| (rows.to_vec(), entries.to_vec());
    let views = (0..nlevels)
        .map(|l| {
            let lists = plan.level(l);
            [
                csr(lists.p2m().row_offsets(), lists.p2m().entries()),
                csr(lists.x().row_offsets(), lists.x().entries()),
                csr(lists.l2p().row_offsets(), lists.l2p().entries()),
                csr(lists.w().row_offsets(), lists.w().entries()),
            ]
        })
        .collect();
    let box_keys = (0..nlevels).map(|l| index.keys(l).to_vec()).collect();
    let leaf_keys: Vec<u64> = index.leaves().keys().to_vec();
    let leaf_levels: Vec<usize> = index
        .leaves()
        .levels()
        .iter()
        .map(|&l| usize::from(l))
        .collect();
    let local = (0..nlevels).map(|l| index.leaves().local(l)).collect();
    let mut rng = SplitMix64::new(0x7e_7001);
    let mut uniform = || rng.range(-1.0, 1.0) as f32;
    let scan = |counts: &[usize]| -> Vec<usize> {
        std::iter::once(0)
            .chain(counts.iter().scan(0, |total, &n| {
                *total += n;
                Some(*total)
            }))
            .collect()
    };
    let source_offsets = scan(fmm.source_counts());
    let target_offsets = scan(fmm.target_counts());
    let mut sources = Vec::with_capacity(4 * source_offsets[source_offsets.len() - 1]);
    for &n in fmm.source_counts() {
        sources.extend((0..4 * n).map(|_| uniform()));
    }
    let targets: Vec<f32> = (0..3 * target_offsets[target_offsets.len() - 1])
        .map(|_| uniform())
        .collect();
    let n = (p + 1) * (p + 1);
    let coefficients = (0..nlevels)
        .map(|l| {
            (0..index.len(l) * n)
                .map(|i| uniform() / (1.0 + (i % n) as f32))
                .collect()
        })
        .collect();
    Data {
        nlevels,
        views,
        box_keys,
        leaf_keys,
        leaf_levels,
        local,
        sources,
        source_offsets,
        targets,
        target_offsets,
        coefficients,
    }
}

/// The level and index of the Morton key `key` on `level`: the integer centre on its own
/// level is 2 i + 1 (CONVENTIONS §3.13, "Integer centres").
fn decode(key: u64, level: usize) -> (u32, [u32; 3]) {
    let centre = nd_fmm_exec::geometry::integer_centre(key, level);
    (level as u32, centre.map(|c| ((c - 1) / 2) as u32))
}

/// Times every (operator, level) of `data`.
fn measure(
    device: &mut Device,
    (problem, p): (&'static str, usize),
    data: &Data,
    host: &LaplaceOperator<f32>,
    threads: usize,
) -> Vec<Row> {
    let n = (p + 1) * (p + 1);
    let leaves: Vec<(u32, [u32; 3])> = data
        .leaf_keys
        .iter()
        .zip(&data.leaf_levels)
        .map(|(&k, &l)| decode(k, l))
        .collect();
    let levels: Vec<Vec<[u32; 3]>> = data
        .box_keys
        .iter()
        .enumerate()
        .map(|(l, keys)| keys.iter().map(|&k| decode(k, l).1).collect())
        .collect();
    let to_u32 = |v: &[usize]| -> Vec<u32> { v.iter().map(|&x| x as u32).collect() };
    let leaf_coordinates = LeafCoordinates::upload(device, &leaves).unwrap();
    let box_coordinates = BoxCoordinates::upload(device, &levels).unwrap();
    let source_offsets = PointOffsets::upload(device, &to_u32(&data.source_offsets)).unwrap();
    let target_offsets = PointOffsets::upload(device, &to_u32(&data.target_offsets)).unwrap();
    let sources = device.upload(&data.sources).unwrap();
    let target_input = device.upload(&data.targets).unwrap();
    let total_targets = data.target_offsets[data.target_offsets.len() - 1];
    let mut output = device.alloc::<f32>(4 * total_targets).unwrap();
    let coefficients: Vec<DeviceBuffer<f32>> = data
        .coefficients
        .iter()
        .map(|c| device.upload(c).unwrap())
        .collect();
    let mut layouts = vec![LeafLayout::default_for(device.info(), p, Precision::F32)];
    if device.backend().is_gpu() {
        layouts.push(LeafLayout::Cube {
            units: 32,
            tile: 16,
        });
    }
    let mut rows = Vec::new();
    for level in 0..data.nlevels {
        for kind in Kind::ALL {
            let (row_offsets, entries) = &data.views[level][kind as usize];
            if entries.is_empty() {
                continue;
            }
            let entry_level = level + usize::from(kind == Kind::M2p);
            let columns = match kind {
                Kind::P2m | Kind::P2l => leaves.len(),
                Kind::L2p | Kind::M2p => levels[entry_level].len(),
            };
            let view = IndexView::upload(device, row_offsets, entries, columns).unwrap();
            let nrows = row_offsets.len() - 1;
            let points: usize = match kind {
                Kind::P2m | Kind::P2l => entries
                    .iter()
                    .map(|&j| data.source_offsets[j as usize + 1] - data.source_offsets[j as usize])
                    .sum(),
                Kind::L2p | Kind::M2p => {
                    let local = &data.local[level];
                    data.target_offsets[local.end] - data.target_offsets[local.start]
                }
            };
            eprintln!("{problem}, p = {p}: {} on level {level}", kind.name());
            // The host result and times.
            let (host_out, host_one) = host_level(kind, (level, p), data, host, 1);
            let (_, host_many) = host_level(kind, (level, p), data, host, threads);
            let mut timed = Vec::new();
            for &layout in &layouts {
                let launch = |device: &mut Device, out: &mut DeviceBuffer<f32>| match kind {
                    Kind::P2m | Kind::P2l => {
                        let inputs = SourceInputs {
                            view: &view,
                            level,
                            boxes: &box_coordinates,
                            leaves: &leaf_coordinates,
                            source_offsets: &source_offsets,
                            sources: sources.as_slice(),
                        };
                        let out = out.slice_mut(..levels[level].len() * n);
                        if kind == Kind::P2m {
                            p2m(device, layout, p, &inputs, out).unwrap();
                        } else {
                            p2l(device, layout, p, &inputs, out).unwrap();
                        }
                    }
                    Kind::L2p | Kind::M2p => {
                        let inputs = TargetInputs {
                            view: &view,
                            level,
                            first_leaf: data.local[level].start,
                            boxes: &box_coordinates,
                            leaves: &leaf_coordinates,
                            target_offsets: &target_offsets,
                            target_input: target_input.as_slice(),
                        };
                        let c = coefficients[entry_level].as_slice();
                        if kind == Kind::L2p {
                            l2p(device, layout, p, true, &inputs, c, out.as_slice_mut()).unwrap();
                        } else {
                            m2p(device, layout, p, true, &inputs, c, out.as_slice_mut()).unwrap();
                        }
                    }
                };
                // From zero: the result against the host's, then the warm-up is done.
                let len = match kind {
                    Kind::P2m | Kind::P2l => levels[level].len() * n,
                    Kind::L2p | Kind::M2p => 4 * total_targets,
                };
                let mut fresh = device.alloc::<f32>(len.max(1)).unwrap();
                let target = if matches!(kind, Kind::L2p | Kind::M2p) {
                    &mut output
                } else {
                    &mut fresh
                };
                let zeros = vec![0.0f32; target.len()];
                device.write(target.as_slice_mut(), &zeros).unwrap();
                launch(device, target);
                let mut got = vec![0.0f32; target.len()];
                device.download(target.as_slice(), &mut got).unwrap();
                let difference = relative_l2(&got[..host_out.len()], &host_out);
                let seconds = median_time_per_call(|calls| {
                    let start = Instant::now();
                    for _ in 0..calls {
                        launch(device, target);
                    }
                    device.sync().unwrap();
                    start.elapsed()
                });
                timed.push((layout, seconds, difference));
            }
            rows.push(Row {
                problem,
                p,
                kind,
                level,
                rows: nrows,
                entries: entries.len(),
                points,
                device: timed,
                host_one,
                host_many,
            });
        }
    }
    rows
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

/// The host operator's level call of `kind` on `threads` scoped threads, each a
/// contiguous share of the rows: its output from zero (the level's coefficients, or the
/// whole target output store) and its median time per call.
fn host_level(
    kind: Kind,
    (level, p): (usize, usize),
    data: &Data,
    host: &LaplaceOperator<f32>,
    threads: usize,
) -> (Vec<f32>, f64) {
    let n = (p + 1) * (p + 1);
    let (row_offsets, entries) = &data.views[level][kind as usize];
    let nrows = row_offsets.len() - 1;
    let row = |r: usize| &entries[row_offsets[r] as usize..row_offsets[r + 1] as usize];
    let source =
        |j: usize| &data.sources[4 * data.source_offsets[j]..4 * data.source_offsets[j + 1]];
    let local = data.local[level].clone();
    let total_targets = data.target_offsets[data.target_offsets.len() - 1];
    let len = match kind {
        Kind::P2m | Kind::P2l => data.box_keys[level].len() * n,
        Kind::L2p | Kind::M2p => 4 * total_targets,
    };
    // The output offset of row r.
    let out_start = |r: usize| match kind {
        Kind::P2m | Kind::P2l => r * n,
        Kind::L2p | Kind::M2p => 4 * data.target_offsets[local.start + r],
    };
    let shares: Vec<std::ops::Range<usize>> = (0..threads)
        .map(|k| k * nrows / threads..(k + 1) * nrows / threads)
        .collect();
    let mut operators: Vec<LaplaceOperator<f32>> = (0..threads).map(|_| host.clone()).collect();
    let mut out = vec![0.0f32; len];
    let run = |out: &mut [f32], operators: &mut [LaplaceOperator<f32>]| {
        // Split the output at the share boundaries.
        let mut rest = &mut out[..];
        let mut consumed = 0;
        let mut pieces = Vec::with_capacity(threads);
        for share in &shares {
            let end = if share.end == nrows {
                match kind {
                    Kind::P2m | Kind::P2l => nrows * n,
                    Kind::L2p | Kind::M2p => 4 * data.target_offsets[local.end],
                }
            } else {
                out_start(share.end)
            };
            let begin = out_start(share.start).max(consumed);
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(begin - consumed);
            let (piece, tail) = tail.split_at_mut(end.max(begin) - begin);
            pieces.push((share.clone(), begin, piece));
            rest = tail;
            consumed = end.max(begin);
        }
        std::thread::scope(|scope| {
            for ((share, begin, piece), op) in pieces.into_iter().zip(operators.iter_mut()) {
                scope.spawn(move || {
                    for r in share {
                        for &e in row(r) {
                            let e = e as usize;
                            match kind {
                                Kind::P2m => {
                                    let at = r * n - begin;
                                    op.p2m_leaf(source(e), &mut piece[at..at + n]);
                                }
                                Kind::P2l => {
                                    let at = r * n - begin;
                                    let target = data.box_keys[level][r];
                                    op.p2l_pair(
                                        data.leaf_keys[e],
                                        target,
                                        source(e),
                                        &mut piece[at..at + n],
                                    );
                                }
                                Kind::L2p | Kind::M2p => {
                                    let leaf = local.start + r;
                                    let (a, b) =
                                        (data.target_offsets[leaf], data.target_offsets[leaf + 1]);
                                    let input = &data.targets[3 * a..3 * b];
                                    let at = 4 * a - begin;
                                    let output = &mut piece[at..at + 4 * (b - a)];
                                    if kind == Kind::L2p {
                                        let c = &data.coefficients[level][e * n..(e + 1) * n];
                                        op.l2p_leaf(c, input, output);
                                    } else {
                                        let c = &data.coefficients[level + 1][e * n..(e + 1) * n];
                                        op.m2p_pair(
                                            data.box_keys[level + 1][e],
                                            data.leaf_keys[leaf],
                                            c,
                                            input,
                                            output,
                                        );
                                    }
                                }
                            }
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

fn print_table(rows: &[Row], device: &Device, threads: usize) {
    let Some(first) = rows.first() else {
        return;
    };
    println!("## {}, p = {}, f32, gradients", first.problem, first.p);
    println!();
    let layouts: Vec<String> = first.device.iter().map(|d| d.0.to_string()).collect();
    print!("| operator | level | rows | entries | points |");
    for layout in &layouts {
        print!(" {} µs | difference |", layout);
    }
    println!(" host 1 thread µs | host {threads} threads µs | x host 1 | x host {threads} |");
    print!("| --- | ---: | ---: | ---: | ---: |");
    for _ in &layouts {
        print!(" ---: | ---: |");
    }
    println!(" ---: | ---: | ---: | ---: |");
    let mut totals: Vec<(Kind, Vec<f64>, f64, f64)> = Vec::new();
    for r in rows {
        print!(
            "| {} | {} | {} | {} | {} |",
            r.kind.name(),
            r.level,
            r.rows,
            r.entries,
            r.points
        );
        for &(_, seconds, difference) in &r.device {
            print!(" {:.1} | {:.1e} |", 1e6 * seconds, difference);
        }
        let best = r.device[0].1;
        println!(
            " {:.1} | {:.1} | {:.1} | {:.2} |",
            1e6 * r.host_one,
            1e6 * r.host_many,
            r.host_one / best,
            r.host_many / best
        );
        match totals.iter_mut().find(|t| t.0 == r.kind) {
            Some(t) => {
                for (s, d) in t.1.iter_mut().zip(&r.device) {
                    *s += d.1;
                }
                t.2 += r.host_one;
                t.3 += r.host_many;
            }
            None => totals.push((
                r.kind,
                r.device.iter().map(|d| d.1).collect(),
                r.host_one,
                r.host_many,
            )),
        }
    }
    println!();
    println!(
        "Totals over the levels ({}; the speed-ups of the first layout, {}):",
        device.info().backend.name(),
        layouts[0]
    );
    println!();
    print!("| operator |");
    for layout in &layouts {
        print!(" {} ms |", layout);
    }
    println!(" host 1 thread ms | host {threads} threads ms | x host 1 | x host {threads} |");
    print!("| --- |");
    for _ in &layouts {
        print!(" ---: |");
    }
    println!(" ---: | ---: | ---: | ---: |");
    for (kind, device_times, one, many) in &totals {
        print!("| {} |", kind.name());
        for s in device_times {
            print!(" {:.3} |", 1e3 * s);
        }
        println!(
            " {:.3} | {:.3} | {:.1} | {:.2} |",
            1e3 * one,
            1e3 * many,
            one / device_times[0],
            many / device_times[0]
        );
    }
    println!();
}

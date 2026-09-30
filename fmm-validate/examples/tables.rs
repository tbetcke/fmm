//! Tables report: what the operator tables of `nd-fmm-tables` cost and what they buy.
//! Build time per family, memory per family and form, time per application against
//! the `nd-fmm-ref` operators, the cache cold against warm, and a recommendation for
//! the per-pair CPU M2L of Phase 3. Prints Markdown on stdout.
//!
//! Run in release mode (a few minutes; building the dense M2L table at p = 20 alone
//! takes tens of seconds):
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example tables
//! ```
//!
//! Everything runs in f64 on one thread, and every table is built once per degree.
//! Each time per application is the median over repeated batches, timed with
//! `std::time::Instant` as in the `timing` example (`nd_fmm_validate::bench`). One
//! call of a batch is a pass over every entry of the family in table order, the 316
//! V-list offsets for M2L or the 8 child octants for M2M and L2L, so that each figure
//! is the mean over the family and a table is read as a per-pair FMM reads it; one
//! extra column applies the dense M2L of a single offset, which stays in cache. The
//! `nd-fmm-ref` operators run at the canonical frames of CONVENTIONS §3.12, and the
//! rotation operator includes building its blocks, as on every call.
//!
//! The cache is timed in a new directory next to the example binary under `target/`,
//! which the example removes afterwards. Timings vary with the machine and its load;
//! this example only reports them and asserts nothing.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use nd_fmm_math::Layout;
use nd_fmm_ref::{Frame, Workspace, direct, rotation};
use nd_fmm_tables::cache::{CacheKey, CacheOutcome, CachedTable, TableCache};
use nd_fmm_tables::geometry::{
    M2L_OFFSET_COUNT, OCTANT_COUNT, l2l_frames, m2l_frames, m2l_offset_index, m2m_frames,
};
use nd_fmm_tables::rotation::Operator;
use nd_fmm_tables::symmetry::SignedPermutation;
use nd_fmm_tables::{
    L2lTables, M2lClasses, M2lScratch, M2lTables, M2mTables, RotationScratch, RotationTables,
};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{
    BATCHES, Crossover, MIN_BATCH, cores, cpu_model, crossover, fitted_exponent,
    median_time_per_call, target,
};

/// Degrees at which every family is built and its memory reported.
const BUILD_PS: [usize; 5] = [4, 8, 12, 16, 20];

/// Extra degree for the octant tables (M2M and L2L) only.
const OCTANT_EXTRA: usize = 30;

/// Degrees at which the applications are timed.
const APPLY_PS: [usize; 8] = [2, 4, 6, 8, 10, 12, 16, 20];

/// Smallest degree used in the fit of the exponent.
const FIT_FROM: usize = 8;

/// Degrees at which the cache is timed.
const CACHE_PS: [usize; 2] = [8, 16];

/// The offset of the single-offset dense M2L column: that of the `timing` example.
const ONE_OFFSET: [i64; 3] = [3, -2, 1];

/// The M2L methods, in the order of the timing columns.
const M2L_METHODS: [&str; 6] = [
    "dense",
    "dense, one offset",
    "classes",
    "table rotation",
    "ref rotation",
    "ref direct",
];

/// The M2M and L2L methods, in the order of the timing columns.
const OCTANT_METHODS: [&str; 4] = ["dense", "table rotation", "ref rotation", "ref direct"];

/// Build times in seconds of one degree; `None` where the family is not built.
#[derive(Clone, Copy, Default)]
struct BuildTimes {
    octant: f64,
    m2l: Option<f64>,
    classes: Option<f64>,
    rotation: Option<f64>,
}

/// Stored reals of each family and form at one degree; `None` where not built.
#[derive(Clone, Copy, Default)]
struct Memory {
    m2m: usize,
    l2l: usize,
    m2l: Option<usize>,
    classes: Option<usize>,
    rotation: Option<[usize; 3]>,
}

/// Times per application in seconds at one degree.
struct ApplyTimes {
    m2l: [f64; 6],
    m2m: [f64; 4],
    l2l: [f64; 4],
}

/// The timing table of one operator: its name, the method names, and the times of
/// each method at the degrees of [`APPLY_PS`].
type Section = (&'static str, &'static [&'static str], Vec<Vec<f64>>);

/// Seconds taken by `f`, and its result.
fn timed<R>(f: impl FnOnce() -> R) -> (R, f64) {
    let start = Instant::now();
    let r = f();
    (r, start.elapsed().as_secs_f64())
}

/// Median time of one application, averaged over a pass over `entries` entries:
/// `apply(e, output)` adds the translation of entry `e` to `output`, (p + 1)² reals.
fn per_application(n: usize, entries: usize, mut apply: impl FnMut(usize, &mut [f64])) -> f64 {
    let mut output = vec![0.0; n];
    let per_pass = median_time_per_call(|passes| {
        output.fill(0.0);
        let start = Instant::now();
        for _ in 0..passes {
            for e in 0..entries {
                apply(e, black_box(&mut output));
            }
        }
        let elapsed = start.elapsed();
        black_box(&output);
        elapsed
    });
    per_pass / entries as f64
}

/// The signature shared by `direct::{m2m, l2l, m2l}` and `rotation::{m2m, l2l, m2l}`.
type Translate = fn(usize, &Frame<f64>, &Frame<f64>, &mut Workspace<f64>, &[f64], &mut [f64]);

/// Time per application of an nd-fmm-ref operator, over a pass over `frames`.
fn reference(p: usize, f: Translate, frames: &[(Frame<f64>, Frame<f64>)], input: &[f64]) -> f64 {
    let mut ws = Workspace::new(p);
    per_application(input.len(), frames.len(), |e, out| {
        let (from, to) = &frames[e];
        f(p, from, to, &mut ws, black_box(input), out);
    })
}

/// Times every method at degree `p`.
fn time_applications(
    p: usize,
    tables: (&M2mTables<f64>, &L2lTables<f64>, &M2lTables<f64>),
    classes: &M2lClasses<f64>,
    rot: &RotationTables<f64>,
) -> ApplyTimes {
    let (m2m, l2l, m2l) = tables;
    let n = Layout::new(p).len();
    let mut rng = SplitMix64::new(p as u64);
    let input: Vec<f64> = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    let x = &input[..];
    let mut m2l_scratch = M2lScratch::new(p);
    let mut rot_scratch = RotationScratch::new(p);
    let one = m2l_offset_index(ONE_OFFSET).expect("a V-list offset");

    let m2l_frames: Vec<_> = (0..M2L_OFFSET_COUNT).map(m2l_frames).collect();
    let m2m_frames: Vec<_> = (0..OCTANT_COUNT).map(m2m_frames).collect();
    let l2l_frames: Vec<_> = (0..OCTANT_COUNT).map(l2l_frames).collect();
    let (d, o) = (M2L_OFFSET_COUNT, OCTANT_COUNT);

    ApplyTimes {
        m2l: [
            per_application(n, d, |e, out| m2l.apply(e, black_box(x), out)),
            per_application(n, 1, |_, out| m2l.apply(one, black_box(x), out)),
            per_application(n, d, |e, out| {
                classes.apply(e, black_box(x), out, &mut m2l_scratch)
            }),
            per_application(n, d, |e, out| {
                rot.m2l(e, black_box(x), out, &mut rot_scratch)
            }),
            reference(p, rotation::m2l::<f64>, &m2l_frames, x),
            reference(p, direct::m2l::<f64>, &m2l_frames, x),
        ],
        m2m: [
            per_application(n, o, |e, out| m2m.apply(e, black_box(x), out)),
            per_application(n, o, |e, out| {
                rot.m2m(e, black_box(x), out, &mut rot_scratch)
            }),
            reference(p, rotation::m2m::<f64>, &m2m_frames, x),
            reference(p, direct::m2m::<f64>, &m2m_frames, x),
        ],
        l2l: [
            per_application(n, o, |e, out| l2l.apply(e, black_box(x), out)),
            per_application(n, o, |e, out| {
                rot.l2l(e, black_box(x), out, &mut rot_scratch)
            }),
            reference(p, rotation::l2l::<f64>, &l2l_frames, x),
            reference(p, direct::l2l::<f64>, &l2l_frames, x),
        ],
    }
}

/// Stored reals of the class form: the class matrices and T_M(P), T_L(P) of every
/// group element. The 316 (class, element) pairs are indices, not counted.
fn class_reals(classes: &M2lClasses<f64>) -> usize {
    let transforms: usize = SignedPermutation::all()
        .iter()
        .map(|&e| {
            classes.multipole_transform(e).blocks().len()
                + classes.local_transform(e).blocks().len()
        })
        .sum();
    classes.matrices().as_slice().len() + transforms
}

/// A size in bytes, in kB below 1 MB and in MB above (decimal units).
fn size(bytes: usize) -> String {
    let b = bytes as f64;
    if b < 1e6 {
        format!("{:.1} kB", b / 1e3)
    } else {
        format!("{:.1} MB", b / 1e6)
    }
}

/// A time in seconds, in ms below 1 s.
fn seconds(t: f64) -> String {
    if t < 1e-2 {
        format!("{:.2} ms", t * 1e3)
    } else if t < 1.0 {
        format!("{:.1} ms", t * 1e3)
    } else {
        format!("{t:.2} s")
    }
}

/// One row of the cache report.
struct CacheRow {
    family: &'static str,
    p: usize,
    bytes: u64,
    cold: f64,
    cold_outcome: String,
    /// The first load after the store.
    first: f64,
    /// The median of the loads after the first.
    warm: f64,
    warm_outcome: String,
    identical: bool,
}

fn outcome(o: &CacheOutcome) -> String {
    match o {
        CacheOutcome::Loaded => "Loaded".to_string(),
        CacheOutcome::Built => "Built".to_string(),
        CacheOutcome::Rebuilt(e) => format!("Rebuilt ({e})"),
        CacheOutcome::NotStored(e) => format!("NotStored ({e})"),
    }
}

/// Number of warm loads after the first, whose median is reported.
const WARM_LOADS: usize = 5;

/// `load_or_build` of family `F` at degree `p` in an empty cache (cold: build and
/// store), then again, 1 + [`WARM_LOADS`] times (warm: load). The first load after a
/// store is reported apart: it pays a one-off cost of the freshly written file.
fn cold_and_warm<F: CachedTable + PartialEq>(
    cache: &TableCache,
    family: &'static str,
    p: usize,
) -> CacheRow {
    let ((cold_table, cold_outcome), cold) = timed(|| cache.load_or_build::<F>(p));
    let ((warm_table, warm_outcome), first) = timed(|| cache.load_or_build::<F>(p));
    let mut warm: Vec<f64> = (0..WARM_LOADS)
        .map(|_| timed(|| cache.load_or_build::<F>(p)).1)
        .collect();
    warm.sort_by(f64::total_cmp);
    let warm = warm[WARM_LOADS / 2];
    let bytes = std::fs::metadata(cache.path(&CacheKey::of::<F>(p))).map_or(0, |m| m.len());
    CacheRow {
        family,
        p,
        bytes,
        cold,
        cold_outcome: outcome(&cold_outcome),
        first,
        warm,
        warm_outcome: outcome(&warm_outcome),
        identical: cold_table == warm_table,
    }
}

/// A new cache directory next to the example binary, under the target directory:
/// `target/<profile>/tables-example-cache-<process id>`.
fn cache_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("the path of the example binary");
    let profile = exe
        .parent()
        .and_then(|examples| examples.parent())
        .expect("the example binary lies in target/<profile>/examples");
    profile.join(format!("tables-example-cache-{}", std::process::id()))
}

fn main() {
    println!("# Operator tables of nd-fmm-tables: build, memory, application, cache");
    println!();
    println!("- CPU: {}", cpu_model());
    println!("- Cores: {}; everything runs on one thread.", cores());
    println!("- Target: {}", target());
    println!("- f64 unless stated. Tables are built in f64; an f32 table is its rounding.");

    let mut all_ps: Vec<usize> = BUILD_PS.iter().chain(&APPLY_PS).copied().collect();
    all_ps.push(OCTANT_EXTRA);
    all_ps.sort_unstable();
    all_ps.dedup();

    let mut builds: Vec<(usize, BuildTimes)> = Vec::new();
    let mut memory: Vec<(usize, Memory)> = Vec::new();
    let mut applications: Vec<ApplyTimes> = Vec::new();
    for &p in &all_ps {
        eprintln!("p = {p}: building");
        let ((m2m, l2l), octant) = timed(|| (M2mTables::<f64>::build(p), L2lTables::build(p)));
        let (m2m_reals, l2l_reals) = (
            m2m.matrices().as_slice().len(),
            l2l.matrices().as_slice().len(),
        );
        if p == OCTANT_EXTRA {
            builds.push((
                p,
                BuildTimes {
                    octant,
                    ..Default::default()
                },
            ));
            memory.push((
                p,
                Memory {
                    m2m: m2m_reals,
                    l2l: l2l_reals,
                    ..Default::default()
                },
            ));
            continue;
        }
        let (m2l, m2l_time) = timed(|| M2lTables::<f64>::build(p));
        let (classes, classes_time) = timed(|| M2lClasses::<f64>::build(p));
        let (rot, rotation_time) = timed(|| RotationTables::<f64>::build(p));
        if BUILD_PS.contains(&p) {
            builds.push((
                p,
                BuildTimes {
                    octant,
                    m2l: Some(m2l_time),
                    classes: Some(classes_time),
                    rotation: Some(rotation_time),
                },
            ));
            memory.push((
                p,
                Memory {
                    m2m: m2m_reals,
                    l2l: l2l_reals,
                    m2l: Some(m2l.matrices().as_slice().len()),
                    classes: Some(class_reals(&classes)),
                    rotation: Some(
                        [Operator::M2m, Operator::L2l, Operator::M2l]
                            .map(|op| rot.tables(op).storage_len()),
                    ),
                },
            ));
        }
        if APPLY_PS.contains(&p) {
            eprintln!("p = {p}: timing");
            applications.push(time_applications(p, (&m2m, &l2l, &m2l), &classes, &rot));
        }
    }

    report_builds(&builds);
    report_memory(&memory);
    let m2l_crossover = report_applications(&applications);
    report_cache();
    report_recommendation(m2l_crossover, &memory);
}

fn report_builds(builds: &[(usize, BuildTimes)]) {
    println!();
    println!("## Build time");
    println!();
    println!(
        "One build each, serial, in f64 (an f32 table costs the same build plus a cast). \
         \"M2M + L2L\" builds both octant families; \"classes\" is the 16-class M2L form; \
         \"rotation\" is `RotationTables`, all three families."
    );
    println!();
    println!("| p | M2M + L2L | dense M2L | classes | rotation |");
    println!("|---:|---:|---:|---:|---:|");
    let cell = |t: Option<f64>| t.map_or("—".to_string(), seconds);
    for (p, b) in builds {
        println!(
            "| {p} | {} | {} | {} | {} |",
            seconds(b.octant),
            cell(b.m2l),
            cell(b.classes),
            cell(b.rotation)
        );
    }
}

fn report_memory(memory: &[(usize, Memory)]) {
    println!();
    println!("## Memory");
    println!();
    println!(
        "Stored reals of each family and form times 8 bytes (f64) or 4 bytes (f32). \
         \"classes\" counts the 16 class matrices and T_M(P), T_L(P) of the 48 group \
         elements, not the 316 (class, element) index pairs (2.5 kB). The rotation \
         families count the rotation blocks, azimuth factors and coaxial factors \
         (`ShiftTables::storage_len`), not their f64 angles, distances and per-entry \
         shifts (a few kB)."
    );
    for (precision, bytes) in [("f64", 8), ("f32", 4)] {
        println!();
        println!("### {precision}");
        println!();
        println!(
            "| p | M2M dense | L2L dense | M2L dense | M2L classes | M2M rotation \
             | L2L rotation | M2L rotation |"
        );
        println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
        let cell = |r: Option<usize>| r.map_or("—".to_string(), |r| size(r * bytes));
        for (p, m) in memory {
            let rot = m.rotation.map(|r| r.map(Some)).unwrap_or([None; 3]);
            println!(
                "| {p} | {} | {} | {} | {} | {} | {} | {} |",
                size(m.m2m * bytes),
                size(m.l2l * bytes),
                cell(m.m2l),
                cell(m.classes),
                cell(rot[0]),
                cell(rot[1]),
                cell(rot[2])
            );
        }
    }
}

/// Prints the timing tables, exponents and crossovers; returns the M2L crossover of
/// table-driven rotation against the dense table.
fn report_applications(applications: &[ApplyTimes]) -> Crossover {
    println!();
    println!("## Time per application");
    println!();
    println!(
        "Median over {BATCHES} batches of at least {} ms each, in µs per application. \
         One call of a batch applies every entry of the family once, in table order (316 \
         offsets or 8 octants), and the figure is the mean over the entries. \"dense, one \
         offset\" applies only the offset {ONE_OFFSET:?}, whose matrix stays in cache. \
         \"table rotation\" is `RotationTables`; \"ref rotation\" and \"ref direct\" are \
         `nd_fmm_ref::rotation` (building its blocks on every call) and \
         `nd_fmm_ref::direct` at the canonical frames. \"dense/rot\" is dense time over \
         table-rotation time.",
        MIN_BATCH.as_millis()
    );
    let ps = APPLY_PS;
    let column =
        |f: &dyn Fn(&ApplyTimes) -> f64| -> Vec<f64> { applications.iter().map(f).collect() };
    let sections: [Section; 3] = [
        (
            "M2L",
            &M2L_METHODS,
            (0..6).map(|i| column(&|a| a.m2l[i])).collect(),
        ),
        (
            "M2M",
            &OCTANT_METHODS,
            (0..4).map(|i| column(&|a| a.m2m[i])).collect(),
        ),
        (
            "L2L",
            &OCTANT_METHODS,
            (0..4).map(|i| column(&|a| a.l2l[i])).collect(),
        ),
    ];
    let mut m2l_crossover = Crossover::FirstAtLargest;
    for (name, methods, times) in &sections {
        // The table-rotation column: 3 for M2L, 1 for M2M and L2L.
        let rot = methods
            .iter()
            .position(|&m| m == "table rotation")
            .expect("a table-rotation column");
        println!();
        println!("### {name}");
        println!();
        let mut header = String::from("| p |");
        let mut rule = String::from("|---:|");
        for m in methods.iter() {
            header += &format!(" {m} |");
            rule += "---:|";
        }
        println!("{header} dense/rot |");
        println!("{rule}---:|");
        for (i, p) in ps.iter().enumerate() {
            let mut line = format!("| {p} |");
            for t in times {
                line += &format!(" {:.3} |", t[i] * 1e6);
            }
            println!("{line} {:.2} |", times[0][i] / times[rot][i]);
        }
        println!();
        println!("Fitted exponent k of t ∝ pᵏ (least squares in log–log, p ≥ {FIT_FROM}):");
        println!();
        let fits: Vec<String> = methods
            .iter()
            .zip(times)
            .map(|(m, t)| format!("{m} {:.2}", fitted_exponent(&ps, t, FIT_FROM)))
            .collect();
        println!("- {}.", fits.join(", "));
        let dense_vs_rotation = crossover(&ps, &times[0], &times[rot]);
        println!(
            "- Dense table against table rotation: {}.",
            dense_vs_rotation.describe("the dense table", "table rotation")
        );
        if *name == "M2L" {
            m2l_crossover = dense_vs_rotation;
            println!(
                "- Class form against table rotation: {}.",
                crossover(&ps, &times[2], &times[rot]).describe("the class form", "table rotation")
            );
            println!(
                "- Dense table against the class form: {}.",
                crossover(&ps, &times[0], &times[2]).describe("the dense table", "the class form")
            );
        }
    }
    m2l_crossover
}

fn report_cache() {
    let dir = cache_dir();
    // A directory left by a crashed run of the same process id would hold warm files.
    let _ = std::fs::remove_dir_all(&dir);
    let cache = TableCache::new(&dir);
    let mut rows = Vec::new();
    for p in CACHE_PS {
        eprintln!("p = {p}: cache");
        rows.push(cold_and_warm::<M2mTables<f64>>(&cache, "M2M", p));
        rows.push(cold_and_warm::<L2lTables<f64>>(&cache, "L2L", p));
        rows.push(cold_and_warm::<M2lTables<f64>>(&cache, "M2L dense", p));
        rows.push(cold_and_warm::<M2lClasses<f64>>(&cache, "M2L classes", p));
        rows.push(cold_and_warm::<RotationTables<f64>>(&cache, "rotation", p));
    }
    let removed = std::fs::remove_dir_all(&dir);

    println!();
    println!("## Cache");
    println!();
    println!(
        "`TableCache::load_or_build` in an empty directory (cold: build and store, one \
         run), then again (warm: load). \"first load\" is the first warm load, which \
         pays a one-off cost for the freshly written file; \"warm\" is the median of the \
         {WARM_LOADS} loads after it, and \"cold/warm\" and \"load rate\" use it. f64. \
         The directory was {} and {}.",
        dir.display(),
        match removed {
            Ok(()) => "has been removed".to_string(),
            Err(e) => format!("could NOT be removed ({e})"),
        }
    );
    println!();
    println!(
        "| family | p | file | cold | outcome | first load | warm | outcome | cold/warm \
         | load rate | warm = cold |"
    );
    println!("|:---|---:|---:|---:|:---|---:|---:|:---|---:|---:|:---:|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.1} | {:.2} GB/s | {} |",
            r.family,
            r.p,
            size(r.bytes as usize),
            seconds(r.cold),
            r.cold_outcome,
            seconds(r.first),
            seconds(r.warm),
            r.warm_outcome,
            r.cold / r.warm,
            r.bytes as f64 / r.warm / 1e9,
            if r.identical { "yes" } else { "NO" }
        );
    }
}

fn report_recommendation(m2l: Crossover, memory: &[(usize, Memory)]) {
    println!();
    println!("## Recommendation: per-pair CPU M2L in Phase 3");
    println!();
    let choice = match m2l {
        Crossover::SecondAlways => {
            "table-driven rotation (`RotationTables::m2l`) at every measured p".to_string()
        }
        Crossover::FirstAtLargest => {
            "the dense table (`M2lTables::apply`) at every measured p (p ≤ 20)".to_string()
        }
        Crossover::From {
            second_from,
            first_at,
        } => format!(
            "the dense table (`M2lTables::apply`) for p ≤ {first_at} and table-driven \
             rotation (`RotationTables::m2l`) for p ≥ {second_from}; between the two \
             measured degrees either is close"
        ),
    };
    println!("- Measured: {choice}.");
    let dense: Vec<String> = memory
        .iter()
        .filter_map(|(p, m)| m.m2l.map(|r| format!("{} at p = {p}", size(r * 8))))
        .collect();
    println!(
        "- Memory per rank of the dense M2L table in f64: {}; the rotation tables and the \
         class form stay below a few tens of MB at p ≤ 20 (Memory above).",
        dense.join(", ")
    );
    println!(
        "- This holds for single-threaded per-pair application only: one matrix–vector \
         product or one rotation–translation–rotation per (source, target) pair, as the \
         host path of C3.1 applies them. The batched GEMM path of Phase 4 applies one \
         table to thousands of multipoles at once, at high arithmetic intensity, and \
         changes the comparison."
    );
}

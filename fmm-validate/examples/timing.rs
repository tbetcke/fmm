//! Timing report: time per call of the direct O(p⁴) and rotation-based O(p³)
//! translations M2M, L2L and M2L of `nd-fmm-ref`, a fitted exponent for p ≥ 8 and the
//! crossover p above which rotation is faster. Prints Markdown on stdout.
//!
//! Run in release mode:
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example timing
//! ```
//!
//! Each figure is the median over repeated batches, timed with `std::time::Instant`;
//! a batch repeats one call until it takes at least 20 ms. The benchmark is
//! single-threaded, runs in f64 and includes everything an operator call does (for
//! rotation, building the rotation blocks). Timings vary with the machine and its
//! load; this example only reports them and asserts nothing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use nd_fmm_ref::{Frame, Workspace, direct, rotation};
use nd_fmm_validate::SplitMix64;

/// Degrees timed for every operator.
const PS: [usize; 8] = [2, 4, 6, 8, 10, 12, 16, 20];

/// Extra degree timed for M2M and L2L only (M2L is tested up to p = 20,
/// CONVENTIONS §3.9).
const P_EXTRA: usize = 30;

/// Smallest degree used in the fit of the exponent.
const FIT_FROM: usize = 8;

/// Number of timed batches per figure; the median is reported.
const BATCHES: usize = 15;

/// Minimum duration of one batch.
const MIN_BATCH: Duration = Duration::from_millis(20);

/// The signature shared by `direct::{m2m, l2l, m2l}` and `rotation::{m2m, l2l, m2l}`.
type Translate = fn(usize, &Frame<f64>, &Frame<f64>, &mut Workspace<f64>, &[f64], &mut [f64]);

#[derive(Clone, Copy)]
enum Op {
    M2m,
    L2l,
    M2l,
}

impl Op {
    const ALL: [Op; 3] = [Op::M2m, Op::L2l, Op::M2l];

    fn name(self) -> &'static str {
        match self {
            Op::M2m => "M2M",
            Op::L2l => "L2L",
            Op::M2l => "M2L",
        }
    }

    /// Direct and rotation implementations.
    fn functions(self) -> [Translate; 2] {
        match self {
            Op::M2m => [direct::m2m::<f64>, rotation::m2m::<f64>],
            Op::L2l => [direct::l2l::<f64>, rotation::l2l::<f64>],
            Op::M2l => [direct::m2l::<f64>, rotation::m2l::<f64>],
        }
    }

    /// Input and output frames of a typical octree use, with a shift off the z axis
    /// so that the rotation operators run all three steps: child (octant (+, −, +))
    /// to parent for M2M, parent to that child for L2L, and the V-list offset
    /// (3, −2, 1) on one level for M2L.
    fn frames(self) -> (Frame<f64>, Frame<f64>) {
        let parent = Frame::new([0.5, 0.5, 0.5], 0.5);
        let child = Frame::new([0.75, 0.25, 0.75], 0.25);
        let neighbour = Frame::new([3.5, -1.5, 1.5], 0.5);
        match self {
            Op::M2m => (child, parent),
            Op::L2l => (parent, child),
            Op::M2l => (parent, neighbour),
        }
    }

    fn degrees(self) -> Vec<usize> {
        let mut ps = PS.to_vec();
        if !matches!(self, Op::M2l) {
            ps.push(P_EXTRA);
        }
        ps
    }
}

/// Median time per call, in seconds, of `f` at degree `p`.
fn time_per_call(f: Translate, p: usize, from: &Frame<f64>, to: &Frame<f64>) -> f64 {
    let mut rng = SplitMix64::new(p as u64);
    let len = (p + 1) * (p + 1);
    let input: Vec<f64> = (0..len).map(|_| rng.range(-1.0, 1.0)).collect();
    let mut output = vec![0.0; len];
    let mut ws = Workspace::new(p);
    let mut batch = |calls: usize| {
        output.fill(0.0);
        let start = Instant::now();
        for _ in 0..calls {
            f(
                p,
                from,
                to,
                &mut ws,
                black_box(&input),
                black_box(&mut output),
            );
        }
        let elapsed = start.elapsed();
        black_box(&output);
        elapsed
    };
    // Warm up, and find the number of calls per batch.
    let mut calls = 1;
    while batch(calls) < MIN_BATCH {
        calls *= 2;
    }
    let mut times: Vec<f64> = (0..BATCHES)
        .map(|_| batch(calls).as_secs_f64() / calls as f64)
        .collect();
    times.sort_by(f64::total_cmp);
    times[BATCHES / 2]
}

/// Least-squares slope of ln t against ln p over the points with p ≥ `FIT_FROM`.
fn fitted_exponent(ps: &[usize], times: &[f64]) -> f64 {
    let points: Vec<(f64, f64)> = ps
        .iter()
        .zip(times)
        .filter(|(p, _)| **p >= FIT_FROM)
        .map(|(&p, &t)| ((p as f64).ln(), t.ln()))
        .collect();
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x / n, b + y / n));
    let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    sxy / sxx
}

/// The crossover: the smallest measured p from which rotation is faster at every
/// larger measured p, and the largest measured p at which direct is faster.
fn crossover(ps: &[usize], direct: &[f64], rotation: &[f64]) -> String {
    let last_direct = (0..ps.len()).rev().find(|&i| direct[i] <= rotation[i]);
    match last_direct {
        None => "rotation is faster at every measured p".to_string(),
        Some(i) if i + 1 == ps.len() => "direct is faster at the largest measured p".to_string(),
        Some(i) => format!(
            "rotation is faster for p ≥ {} (direct is faster at p = {})",
            ps[i + 1],
            ps[i]
        ),
    }
}

/// The value of a `sysctl` key (macOS), if it can be read.
fn sysctl(key: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    (out.status.success() && !s.is_empty()).then(|| s.to_string())
}

/// The CPU model, from `sysctl` on macOS and `/proc/cpuinfo` on Linux.
fn cpu_model() -> String {
    let from_proc = || {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines()
            .find(|l| l.starts_with("model name") || l.starts_with("Model"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
    };
    let model = if cfg!(target_os = "macos") {
        sysctl("machdep.cpu.brand_string")
    } else {
        from_proc()
    };
    model.unwrap_or_else(|| "unknown".to_string())
}

/// The core count: logical CPUs from `std::thread::available_parallelism`, and on
/// macOS the physical cores from `sysctl`.
fn cores() -> String {
    let logical = std::thread::available_parallelism().map_or(0, |n| n.get());
    match sysctl("hw.physicalcpu") {
        Some(physical) if cfg!(target_os = "macos") => {
            format!("{physical} physical, {logical} logical")
        }
        _ => format!("{logical} logical"),
    }
}

fn main() {
    println!("# Direct versus rotation translations in nd-fmm-ref");
    println!();
    println!("- CPU: {}", cpu_model());
    println!("- Cores: {}; the benchmark runs on one thread.", cores());
    println!(
        "- Target: {}-{}{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(debug_assertions) {
            " (DEBUG BUILD: timings are not meaningful; use --release)"
        } else {
            ", release build"
        }
    );
    println!(
        "- f64; median over {BATCHES} batches of at least {} ms each, time per call in µs. \
         M2M child to parent, L2L parent to child (octant (+, −, +)), M2L for the V-list \
         offset (3, −2, 1). \"d/r\" is direct time over rotation time.",
        MIN_BATCH.as_millis()
    );
    println!();

    // times[op][method][i] for the degrees of `op`.
    let times: Vec<[Vec<f64>; 2]> = Op::ALL
        .iter()
        .map(|&op| {
            let (from, to) = op.frames();
            op.functions().map(|f| {
                op.degrees()
                    .iter()
                    .map(|&p| time_per_call(f, p, &from, &to))
                    .collect()
            })
        })
        .collect();

    let mut header = String::from("| p |");
    let mut rule = String::from("|---:|");
    for op in Op::ALL {
        let name = op.name();
        header += &format!(" {name} direct | {name} rotation | {name} d/r |");
        rule += "---:|---:|---:|";
    }
    println!("{header}");
    println!("{rule}");
    let all_ps: Vec<usize> = PS.iter().copied().chain([P_EXTRA]).collect();
    for p in all_ps {
        let mut line = format!("| {p} |");
        for (o, op) in Op::ALL.iter().enumerate() {
            match op.degrees().iter().position(|&q| q == p) {
                Some(i) => {
                    let [d, r] = [times[o][0][i], times[o][1][i]];
                    line += &format!(" {:.2} | {:.2} | {:.2} |", d * 1e6, r * 1e6, d / r);
                }
                None => line += " — | — | — |",
            }
        }
        println!("{line}");
    }

    println!();
    println!(
        "Fitted exponent k of t ∝ pᵏ (least squares in log–log, p ≥ {FIT_FROM}), and crossover:"
    );
    println!();
    for (o, op) in Op::ALL.iter().enumerate() {
        let ps = op.degrees();
        let [d, r] = [&times[o][0], &times[o][1]];
        println!(
            "- {}: direct k = {:.2}, rotation k = {:.2}; {}.",
            op.name(),
            fitted_exponent(&ps, d),
            fitted_exponent(&ps, r),
            crossover(&ps, d, r)
        );
    }
}

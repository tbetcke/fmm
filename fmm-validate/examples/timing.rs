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
use std::time::Instant;

use nd_fmm_ref::{Frame, Workspace, direct, rotation};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{
    BATCHES, MIN_BATCH, cores, cpu_model, crossover, fitted_exponent, median_time_per_call, target,
};

/// Degrees timed for every operator.
const PS: [usize; 8] = [2, 4, 6, 8, 10, 12, 16, 20];

/// Extra degree timed for M2M and L2L only (M2L is tested up to p = 20,
/// CONVENTIONS §3.9).
const P_EXTRA: usize = 30;

/// Smallest degree used in the fit of the exponent.
const FIT_FROM: usize = 8;

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
    let batch = |calls: usize| {
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
    median_time_per_call(batch)
}

fn main() {
    println!("# Direct versus rotation translations in nd-fmm-ref");
    println!();
    println!("- CPU: {}", cpu_model());
    println!("- Cores: {}; the benchmark runs on one thread.", cores());
    println!("- Target: {}", target());
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
            fitted_exponent(&ps, d, FIT_FROM),
            fitted_exponent(&ps, r, FIT_FROM),
            crossover(&ps, d, r).describe("direct", "rotation")
        );
    }
}

//! The Helmholtz M2L per pair, Dense against Rotation per degree (Helmholtz CH3.2a;
//! docs/design/helmholtz-fmm-proposal.md §5.3, §10 CH3.2). Prints Markdown on stdout.
//!
//! For each degree p of `--ps` (default 5, 7, 9, 12, 15, 18, 20), one level (level 2 of
//! three, k̂ = k r₂ of `--k`, default 1 and 1 + 0.1i) with:
//!
//! - Rotation: `nd_fmm_tables::helmholtz::HelmholtzTables::m2l` (RCR with the level's
//!   tables, the y-blocks stored);
//! - Dense: `nd_fmm_tables::helmholtz::dense::DenseM2l::m2l` (one complex matrix per offset,
//!   applied pair by pair), its tables' bytes and build seconds;
//!
//! each pass the 316 V-list offsets in turn, each with its own multipole of `--vectors`
//! vectors (default 1) into its own local, as a level call reads the tables; the time per
//! pair is the median over 15 batches of passes (`bench::median_time_per_call`), on one
//! thread. Per real k the real multipoles of real charges (one plane) and complex ones
//! (two planes); per complex k two planes. Before timing, Dense's locals of one pass are
//! compared with Rotation's (the largest relative L2 of a local), so that a timed row is a
//! correct one. The cost per pair does not depend on k (the tables' size and the
//! operations are p's); k only changes the values.
//!
//! Options: `--ps p,…`, `--k re[+im],…` (k̂ on the level, e.g. `1,1+0.1`), `--vectors m`,
//! `--max-mb MB` (skip Dense above this many MB of tables per level; default 2048).
//!
//! Run in release mode with one BLAS thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_m2l -- --ps 5,10,15
//! ```
//!
//! Timings are wall times on this machine, reported and never asserted.

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target};

/// The level of the tables (levels 0 and 1 have no V list).
const LEVEL: usize = 2;

struct Arguments {
    ps: Vec<usize>,
    ks: Vec<Complex<f64>>,
    vectors: usize,
    max_bytes: usize,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_m2l [--ps p,…] [--k re[+im],…] [--vectors m] [--max-mb MB]; \
             got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        ps: vec![5, 7, 9, 12, 15, 18, 20],
        ks: vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.1)],
        vectors: 1,
        max_bytes: 2048 << 20,
    };
    let k = |v: &str| -> Option<Complex<f64>> {
        match v.split_once('+') {
            Some((re, im)) => Some(Complex::new(re.parse().ok()?, im.parse().ok()?)),
            None => Some(Complex::new(v.parse().ok()?, 0.0)),
        }
    };
    for pair in args.chunks(2) {
        let [flag, value] = pair else { usage() };
        let ok = match flag.as_str() {
            "--ps" => value
                .split(',')
                .map(|v| v.parse().ok())
                .collect::<Option<Vec<usize>>>()
                .map(|v| parsed.ps = v)
                .is_some(),
            "--k" => value
                .split(',')
                .map(k)
                .collect::<Option<Vec<_>>>()
                .filter(|ks| ks.iter().all(|k| k.re > 0.0 && k.im >= 0.0))
                .map(|v| parsed.ks = v)
                .is_some(),
            "--vectors" => value
                .parse::<usize>()
                .ok()
                .filter(|&m| m > 0)
                .map(|m| parsed.vectors = m)
                .is_some(),
            "--max-mb" => value
                .parse::<usize>()
                .map(|mb| parsed.max_bytes = mb << 20)
                .is_ok(),
            _ => false,
        };
        if !ok {
            usage();
        }
    }
    parsed
}

/// The 316 multipoles of one pass, uniform in [−1, 1).
fn multipoles(len: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    (0..M2L_OFFSET_COUNT)
        .map(|_| {
            (0..len)
                .map(|_| 2.0 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 1.0)
                .collect()
        })
        .collect()
}

fn main() {
    let arguments = arguments();
    let m = arguments.vectors;
    println!("# Helmholtz M2L per pair: Dense against Rotation (CH3.2a)");
    println!();
    println!(
        "{}, {}, {}; one thread, f64, level {LEVEL}, {m} vector(s), the 316 offsets per pass, \
         OPENBLAS_NUM_THREADS = {}",
        cpu_model(),
        cores(),
        target(),
        std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
    );
    println!();
    println!(
        "| p | k̂ | planes | Rotation ns/pair | Dense ns/pair | Rotation / Dense | Dense MB \
         | Dense build s | Dense against Rotation |"
    );
    println!("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for &p in &arguments.ps {
        for &k_level in &arguments.ks {
            // k̂₀ = 4 k̂₂ (exact): level 2 of three.
            let k_root = k_level * 4.0;
            let degrees = [0, 0, p];
            let tables = HelmholtzTables::<f64>::build(k_root, &degrees, DEFAULT_BLOCK_BUDGET);
            let bytes = DenseM2l::<f64>::level_bytes(p);
            let dense = (bytes <= arguments.max_bytes).then(|| {
                let start = Instant::now();
                let dense = DenseM2l::<f64>::build(k_root, &degrees, &[false, false, true]);
                (dense, start.elapsed().as_secs_f64())
            });
            let n = Layout::new(p).len();
            let plane_counts: &[usize] = if k_level.im == 0.0 { &[1, 2] } else { &[2] };
            for &planes in plane_counts {
                let inputs = multipoles(m * planes * n, p as u64);
                let mut locals = vec![vec![0.0; m * 2 * n]; M2L_OFFSET_COUNT];
                let mut scratch = tables.scratch();
                let mut rotation_pass = |locals: &mut [Vec<f64>]| {
                    for (d, (x, y)) in inputs.iter().zip(locals.iter_mut()).enumerate() {
                        tables.m2l(LEVEL, d, x, planes, y, m, &mut scratch);
                    }
                };
                rotation_pass(&mut locals);
                let reference = locals.clone();
                let rotation = median_time_per_call(|calls| {
                    let start = Instant::now();
                    for _ in 0..calls {
                        rotation_pass(black_box(&mut locals));
                    }
                    start.elapsed()
                }) / M2L_OFFSET_COUNT as f64;
                let k_string = if k_level.im == 0.0 {
                    format!("{}", k_level.re)
                } else {
                    format!("{} + {}i", k_level.re, k_level.im)
                };
                let Some((dense, build)) = &dense else {
                    println!(
                        "| {p} | {k_string} | {planes} | {:.0} | – | – | {:.0} (skipped) | – | – |",
                        rotation * 1e9,
                        bytes as f64 / 1e6
                    );
                    continue;
                };
                let dense_pass = |locals: &mut [Vec<f64>]| {
                    for (d, (x, y)) in inputs.iter().zip(locals.iter_mut()).enumerate() {
                        dense.m2l(LEVEL, d, x, planes, y, m);
                    }
                };
                for local in &mut locals {
                    local.fill(0.0);
                }
                dense_pass(&mut locals);
                let worst = locals
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| {
                        let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
                        let norm: f64 = b.iter().map(|y| y * y).sum();
                        (diff / norm).sqrt()
                    })
                    .fold(0.0f64, f64::max);
                let dense_time = median_time_per_call(|calls| {
                    let start = Instant::now();
                    for _ in 0..calls {
                        dense_pass(black_box(&mut locals));
                    }
                    start.elapsed()
                }) / M2L_OFFSET_COUNT as f64;
                println!(
                    "| {p} | {k_string} | {planes} | {:.0} | {:.0} | {:.2} | {:.0} | {build:.2} \
                     | {worst:.1e} |",
                    rotation * 1e9,
                    dense_time * 1e9,
                    rotation / dense_time,
                    bytes as f64 / 1e6,
                );
            }
        }
    }
}

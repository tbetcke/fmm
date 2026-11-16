//! The Helmholtz RCR M2L per pair on one core (Helmholtz CH4.0i; CH4.0e's report §3.2: our
//! pair 1.9–3.3× S6's prototype on the same core). Prints Markdown on stdout.
//!
//! For each `p@k̂` of `--cases` (default `72@31.25,131@62.5`: levels 3 and 2 of W1 at
//! kD = 500 at 10⁻⁶, k̂ = k r_l on the level), one level (level 2 of three) of
//! `nd_fmm_tables::helmholtz::HelmholtzTables` at degree p, and per multipole planes (one,
//! the real multipoles of real charges, and two) the time per pair of a pass over the 316
//! V-list offsets, each with its own multipole into its own local:
//!
//! - **stored**: the M2L y-blocks stored (`m2l_blocks_stored`), offsets in index order;
//! - **per call**: the blocks formed per call (a block budget of 0), offsets in index order,
//!   where consecutive offsets seldom share a polar angle;
//! - **by class**: the blocks formed per call, offsets in the order of
//!   `HelmholtzTables::m2l_classes` (angle, then distance), as the pair split of the level
//!   call orders a chunk's pairs: the scratch keeps the last angle's blocks (Helmholtz
//!   CH4.0i), so each angle's are formed once a pass;
//!
//! and `nd_fmm_math::orthonormal::rotation::y_blocks` alone (one angle's blocks at p, 16
//! calls a pass), the share of forming them. Each figure is the median of `--reps` passes (default 3) after one
//! untimed pass; the locals of the variants are compared bit for bit first (they are the
//! same function), so that a timed row is a correct one.
//!
//! Run in release mode with one BLAS thread, on one core of a quiet machine:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_rcr_pair -- --cases 72@31.25
//! ```
//!
//! Timings are wall times on this machine, reported and never asserted.

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_math::orthonormal::rotation::{Scratch, compact_blocks_len, y_blocks};
use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, target};

/// The level of the tables (levels 0 and 1 have no V list).
const LEVEL: usize = 2;

struct Arguments {
    cases: Vec<(usize, f64)>,
    reps: usize,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!("usage: helmholtz_rcr_pair [--cases p@k̂,…] [--reps n]; got {args:?}");
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        cases: vec![(72, 31.25), (131, 62.5)],
        reps: 3,
    };
    for pair in args.chunks(2) {
        let [flag, value] = pair else { usage() };
        let ok = match flag.as_str() {
            "--cases" => value
                .split(',')
                .map(|c| {
                    let (p, k) = c.split_once('@')?;
                    Some((p.parse().ok()?, k.parse::<f64>().ok().filter(|&k| k > 0.0)?))
                })
                .collect::<Option<Vec<_>>>()
                .map(|v| parsed.cases = v)
                .is_some(),
            "--reps" => value
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .map(|n| parsed.reps = n)
                .is_some(),
            _ => false,
        };
        if !ok {
            usage();
        }
    }
    parsed
}

/// The median over `reps` of `pass()`'s seconds, after one untimed pass.
fn median(reps: usize, mut pass: impl FnMut()) -> f64 {
    pass();
    let mut times: Vec<f64> = (0..reps)
        .map(|_| {
            let start = Instant::now();
            pass();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[reps / 2]
}

fn main() {
    let arguments = arguments();
    println!("# Helmholtz RCR M2L per pair (CH4.0i)");
    println!();
    println!(
        "Machine: {}, {}; target {}; one thread; median of {} passes over the 316 offsets.",
        cpu_model(),
        cores(),
        target(),
        arguments.reps
    );
    println!();
    println!(
        "| p | k̂ | planes | stored ms | per call ms | by class ms | per call ÷ stored \
         | by class ÷ stored | y-blocks ms | y-blocks ÷ per call |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for &(p, k_hat) in &arguments.cases {
        // k̂₀ = 4 k̂ on level 2.
        let k_root = Complex::new(4.0 * k_hat, 0.0);
        let degrees = [0, 0, p];
        let stored =
            HelmholtzTables::<f64>::build(k_root, &degrees, DEFAULT_BLOCK_BUDGET.max(1 << 31));
        let per_call = HelmholtzTables::<f64>::build(k_root, &degrees, 0);
        assert!(stored.m2l_blocks_stored(LEVEL) && !per_call.m2l_blocks_stored(LEVEL));
        let len = Layout::new(p).len();
        let index_order: Vec<usize> = (0..M2L_OFFSET_COUNT).collect();
        let mut class_order = index_order.clone();
        class_order.sort_by_key(|&d| per_call.m2l_classes(d));
        // The y-blocks of one angle alone.
        let blocks_ms = {
            let mut out = vec![0.0f64; compact_blocks_len(p)];
            let mut scratch = Scratch::new(p);
            let (c, s) = (0.6f64, 0.8f64);
            // 16 calls a pass: one is short at small p.
            1e3 / 16.0
                * median(arguments.reps, || {
                    for _ in 0..16 {
                        y_blocks(p, black_box(c), black_box(s), &mut out, &mut scratch);
                        black_box(&out);
                    }
                })
        };
        for planes in [1, 2] {
            let mut rng = SplitMix64::new(p as u64 * 31 + planes as u64);
            let multipoles: Vec<Vec<f64>> = (0..M2L_OFFSET_COUNT)
                .map(|_| (0..planes * len).map(|_| rng.uniform() - 0.5).collect())
                .collect();
            let mut locals = vec![vec![0.0f64; 2 * len]; M2L_OFFSET_COUNT];
            let run = |tables: &HelmholtzTables<f64>, order: &[usize], out: &mut Vec<Vec<f64>>| {
                let mut scratch = tables.scratch();
                for &d in order {
                    tables.m2l(
                        LEVEL,
                        d,
                        &multipoles[d],
                        planes,
                        &mut out[d],
                        1,
                        &mut scratch,
                    );
                }
            };
            // The same function: compare one pass of each first.
            let fresh = || vec![vec![0.0f64; 2 * len]; M2L_OFFSET_COUNT];
            let (mut a, mut b, mut c) = (fresh(), fresh(), fresh());
            run(&stored, &index_order, &mut a);
            run(&per_call, &index_order, &mut b);
            run(&per_call, &class_order, &mut c);
            let bits = |x: &Vec<Vec<f64>>| -> Vec<u64> {
                x.iter().flatten().map(|v| v.to_bits()).collect()
            };
            assert!(
                bits(&a) == bits(&b) && bits(&b) == bits(&c),
                "p = {p}: the variants differ"
            );
            let pair = |seconds: f64| 1e3 * seconds / M2L_OFFSET_COUNT as f64;
            let t_stored = pair(median(arguments.reps, || {
                run(&stored, &index_order, &mut locals)
            }));
            let t_call = pair(median(arguments.reps, || {
                run(&per_call, &index_order, &mut locals)
            }));
            let t_class = pair(median(arguments.reps, || {
                run(&per_call, &class_order, &mut locals)
            }));
            black_box(&locals);
            println!(
                "| {p} | {k_hat} | {planes} | {t_stored:.3} | {t_call:.3} | {t_class:.3} | {:.2} \
                 | {:.2} | {blocks_ms:.3} | {:.2} |",
                t_call / t_stored,
                t_class / t_stored,
                blocks_ms / t_call,
            );
        }
    }
}

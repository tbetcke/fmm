//! The Helmholtz M2L per pair, Dense, Classes and Batched against Rotation per degree
//! (Helmholtz CH3.2a, CH3.2b, CH3.2c; docs/design/helmholtz-fmm-proposal.md §5.3, §10 CH3.2).
//! Prints Markdown on stdout.
//!
//! For each degree p of `--ps` (default 5, 7, 9, 12, 15, 18, 20), one level (level 2 of
//! three, k̂ = k r₂ of `--k`, default 1 and 1 + 0.1i) with:
//!
//! - Rotation: `nd_fmm_tables::helmholtz::HelmholtzTables::m2l` (RCR with the level's
//!   tables, the y-blocks stored);
//! - Dense: `nd_fmm_tables::helmholtz::dense::DenseM2l::m2l` (one complex matrix per offset,
//!   applied pair by pair), its tables' bytes and build seconds;
//! - Classes: `nd_fmm_tables::helmholtz::classes::ClassM2l::m2l` (16 class matrices and the
//!   cube-group transforms O(Pᵀ) and O(P) per pair), its tables' bytes (the transforms
//!   included) and build seconds; the time per pair over all 316 offsets (the transforms
//!   that move the z-axis included), over the offsets whose element fixes the z-axis
//!   (signed permutations) and over the others (dense per-degree blocks), and the
//!   transforms alone (O(Pᵀ) on each multipole plane and O(P) on both local planes, over
//!   all 316 offsets, without `m2l`'s two scratch fills);
//! - Batched: `nd_fmm_tables::helmholtz::batched::BatchedM2l` (Dense's matrices in split
//!   planes) by `nd_fmm_simd::MatrixKernel` (`MulAdd::Separate`, the default ISA), the call
//!   sequence of the Helmholtz operator's batched M2L (its module docs, "M2L": two calls
//!   per offset for one plane, four with L_r negated around the third for two), restated
//!   here: per pass the offsets in turn, each with `--columns` pairs (default 32; a block
//!   of 64 targets on a uniform level has about 38 per offset), their own multipoles into
//!   one block of as many locals, so each table is read once per offset and block; its
//!   build seconds (its bytes are Dense's);
//!
//! the other strategies each pass the offsets in turn, each with its own multipole of
//! `--vectors` vectors (default 1) into its own local, as a level call reads the tables;
//! the time per pair is the median over 15 batches of passes
//! (`bench::median_time_per_call`), on one thread. Per
//! real k the real multipoles of real charges (one plane) and complex ones (two planes); per
//! complex k two planes. Before timing, Dense's and Classes' locals of one pass are compared
//! with Rotation's (the largest relative L2 of a local), and Batched's block with Dense's
//! applied pair by pair to the same pairs, so that a timed row is a correct one. The cost
//! per pair does not depend on k (the tables' size and the operations are p's); k only
//! changes the values.
//!
//! Options: `--ps p,…`, `--k re[+im],…` (k̂ on the level, e.g. `1,1+0.1`), `--vectors m`,
//! `--columns c` (Batched's pairs per offset), `--max-mb MB` (skip Dense and Batched above
//! this many MB of tables per level; default 2048).
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
use nd_fmm_simd::{MatrixKernel, MulAdd};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::ClassM2l;
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_tables::symmetry::class_of;
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, median_time_per_call, target};

/// The level of the tables (levels 0 and 1 have no V list).
const LEVEL: usize = 2;

struct Arguments {
    ps: Vec<usize>,
    ks: Vec<Complex<f64>>,
    vectors: usize,
    columns: usize,
    max_bytes: usize,
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_m2l [--ps p,…] [--k re[+im],…] [--vectors m] [--columns c] \
             [--max-mb MB]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        ps: vec![5, 7, 9, 12, 15, 18, 20],
        ks: vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.1)],
        vectors: 1,
        columns: 32,
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
            "--columns" => value
                .parse::<usize>()
                .ok()
                .filter(|&c| c > 0)
                .map(|c| parsed.columns = c)
                .is_some(),
            "--max-mb" => value
                .parse::<usize>()
                .ok()
                .and_then(|mb| mb.checked_mul(1 << 20))
                .map(|bytes| parsed.max_bytes = bytes)
                .is_some(),
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

/// The largest relative L2 of a local of `got` against `reference`.
fn worst_difference(got: &[Vec<f64>], reference: &[Vec<f64>]) -> f64 {
    got.iter()
        .zip(reference)
        .map(|(a, b)| {
            let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            let norm: f64 = b.iter().map(|y| y * y).sum();
            (diff / norm).sqrt()
        })
        .fold(0.0f64, f64::max)
}

/// The Helmholtz operator's batched M2L of one offset (its module docs, "M2L"), restated:
/// the planes (A_r, A_i) on the columns whose real planes start at `sources` in
/// `multipoles` and at `targets` in `locals`.
#[expect(
    clippy::too_many_arguments,
    reason = "the kernel, the planes, two column lists over two slices, and the packing buffer"
)]
fn batched_apply(
    matrix: &MatrixKernel<f64>,
    n: usize,
    planes: usize,
    (a_r, a_i): (&[f64], &[f64]),
    multipoles: &[f64],
    sources: &[usize],
    locals: &mut [f64],
    targets: &[usize],
    pack: &mut [f64],
) {
    let negate = |locals: &mut [f64]| {
        for &o in targets {
            for x in &mut locals[o..o + n] {
                *x = -*x;
            }
        }
    };
    matrix.accumulate_columns(n, a_r, multipoles, sources, locals, targets, pack);
    if planes == 1 {
        matrix.accumulate_columns(n, a_i, multipoles, sources, &mut locals[n..], targets, pack);
        return;
    }
    let imaginary = &multipoles[n..];
    negate(locals);
    matrix.accumulate_columns(n, a_i, imaginary, sources, locals, targets, pack);
    negate(locals);
    let locals_i = &mut locals[n..];
    matrix.accumulate_columns(n, a_i, multipoles, sources, locals_i, targets, pack);
    matrix.accumulate_columns(n, a_r, imaginary, sources, locals_i, targets, pack);
}

/// The median time per pair of `pass` over `pairs` pairs a pass, in ns.
fn ns_per_pair(pairs: usize, mut pass: impl FnMut()) -> f64 {
    median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            pass();
        }
        start.elapsed()
    }) / pairs as f64
        * 1e9
}

fn main() {
    let arguments = arguments();
    let m = arguments.vectors;
    let all: Vec<usize> = (0..M2L_OFFSET_COUNT).collect();
    // Each offset's element P and Pᵀ, formed once.
    let elements: Vec<_> = all
        .iter()
        .map(|&d| {
            let element = class_of(d).1;
            (element, element.transpose())
        })
        .collect();
    let (z_axis, others): (Vec<usize>, Vec<usize>) = all
        .iter()
        .partition(|&&d| elements[d].0.permutation()[2] == 2);
    println!(
        "# Helmholtz M2L per pair: Dense, Classes and Batched against Rotation (CH3.2a, \
         CH3.2b, CH3.2c)"
    );
    println!();
    println!(
        "{}, {}, {}; one thread, f64, level {LEVEL}, {m} vector(s), the 316 offsets per pass \
         ({} whose element fixes the z-axis, {} others), OPENBLAS_NUM_THREADS = {}",
        cpu_model(),
        cores(),
        target(),
        z_axis.len(),
        others.len(),
        std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
    );
    println!();
    println!(
        "ns per pair; Classes over all offsets, over the z-axis offsets (z) and the others \
         (moving), and its two transforms alone (O(Pᵀ) per multipole plane, O(P) per local \
         plane, over all offsets); Batched {} pairs per offset on the {:?} column kernel, \
         its error against Dense applied pair by pair to the same pairs.",
        arguments.columns,
        MatrixKernel::<f64>::detect(MulAdd::Separate).isa(),
    );
    println!();
    println!(
        "| p | k̂ | planes | Rotation | Dense | Classes | Classes z | Classes moving \
         | transforms | Rotation / Classes | Dense / Classes | Dense MB | Classes MB \
         | Dense build s | Classes build s | Dense against Rotation | Classes against Rotation \
         | Batched | Batched fused | Rotation / Batched | Rotation / fused | Dense / Batched \
         | Dense / fused | Batched build s | Batched against Dense | fused against Dense |"
    );
    println!(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: \
         | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: \
         | ---: | ---: |"
    );
    let matrix = MatrixKernel::<f64>::detect(MulAdd::Separate);
    let fused = MatrixKernel::<f64>::detect(MulAdd::Fused);
    let c = arguments.columns;
    for &p in &arguments.ps {
        for &k_level in &arguments.ks {
            // k̂₀ = 4 k̂₂ (exact): level 2 of three.
            let k_root = k_level * 4.0;
            let degrees = [0, 0, p];
            let levels = [false, false, true];
            let tables = HelmholtzTables::<f64>::build(k_root, &degrees, DEFAULT_BLOCK_BUDGET);
            let bytes = DenseM2l::<f64>::level_bytes(p);
            let dense = (bytes <= arguments.max_bytes).then(|| {
                let start = Instant::now();
                let dense = DenseM2l::<f64>::build(k_root, &degrees, &levels);
                (dense, start.elapsed().as_secs_f64())
            });
            let batched = (bytes <= arguments.max_bytes).then(|| {
                let start = Instant::now();
                let batched = BatchedM2l::<f64>::build(k_root, &degrees, &levels);
                (batched, start.elapsed().as_secs_f64())
            });
            let start = Instant::now();
            let classes = ClassM2l::<f64>::build(k_root, &degrees, &levels);
            let classes_build = start.elapsed().as_secs_f64();
            let transforms = classes.transforms().expect("level 2 has classes");
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
                let rotation =
                    ns_per_pair(M2L_OFFSET_COUNT, || rotation_pass(black_box(&mut locals)));

                let mut class_scratch = classes.scratch();
                let mut classes_pass = |offsets: &[usize], locals: &mut [Vec<f64>]| {
                    for &d in offsets {
                        let (x, y) = (&inputs[d], &mut locals[d]);
                        classes.m2l(LEVEL, d, x, planes, y, m, &mut class_scratch);
                    }
                };
                for local in &mut locals {
                    local.fill(0.0);
                }
                classes_pass(&all, &mut locals);
                let classes_worst = worst_difference(&locals, &reference);
                let classes_all =
                    ns_per_pair(all.len(), || classes_pass(&all, black_box(&mut locals)));
                let classes_z = ns_per_pair(z_axis.len(), || {
                    classes_pass(&z_axis, black_box(&mut locals))
                });
                let classes_moving = ns_per_pair(others.len(), || {
                    classes_pass(&others, black_box(&mut locals))
                });
                // The transforms alone: O(Pᵀ) on each multipole plane, O(P) on both local
                // planes (here the multipole's first plane twice), per vector.
                let mut rotated = vec![0.0; n];
                let transforms_only = ns_per_pair(M2L_OFFSET_COUNT, || {
                    for (&(element, transpose), x) in elements.iter().zip(&inputs) {
                        for v in x.chunks_exact(planes * n) {
                            for plane in v.chunks_exact(n) {
                                transforms.apply(transpose, plane, &mut rotated);
                            }
                            for _ in 0..2 {
                                transforms.apply(element, &v[..n], &mut rotated);
                            }
                        }
                        black_box(&mut rotated);
                    }
                });

                let k_string = if k_level.im == 0.0 {
                    format!("{}", k_level.re)
                } else {
                    format!("{} + {}i", k_level.re, k_level.im)
                };
                // Dense's error against Rotation, its time per pair and its build seconds.
                let dense_row = dense.as_ref().map(|(dense, build)| {
                    let dense_pass = |locals: &mut [Vec<f64>]| {
                        for (d, (x, y)) in inputs.iter().zip(locals.iter_mut()).enumerate() {
                            dense.m2l(LEVEL, d, x, planes, y, m);
                        }
                    };
                    for local in &mut locals {
                        local.fill(0.0);
                    }
                    dense_pass(&mut locals);
                    let worst = worst_difference(&locals, &reference);
                    let time = ns_per_pair(M2L_OFFSET_COUNT, || dense_pass(black_box(&mut locals)));
                    (worst, time, *build)
                });
                let (dense_time, dense_ratio, dense_build, dense_worst) = match &dense_row {
                    Some((worst, time, build)) => (
                        format!("{time:.0}"),
                        format!("{:.2}", time / classes_all),
                        format!("{build:.2}"),
                        format!("{worst:.1e}"),
                    ),
                    None => ("–".into(), "–".into(), "–".into(), "–".into()),
                };
                // Batched on a block of c pairs per offset: its time per pair and build
                // seconds, and its error against Dense on the same pairs.
                let batched_row = dense
                    .as_ref()
                    .zip(batched.as_ref())
                    .map(|(dense, batched)| {
                        let (dense, (batched, build)) = (&dense.0, batched);
                        let (input_len, local_len) = (m * planes * n, m * 2 * n);
                        let block_inputs: Vec<f64> = (0..c)
                            .flat_map(|j| inputs[j % M2L_OFFSET_COUNT].iter().copied())
                            .collect();
                        let sources: Vec<usize> = (0..c * m).map(|i| i * planes * n).collect();
                        let targets: Vec<usize> = (0..c * m).map(|i| i * 2 * n).collect();
                        let mut block = vec![0.0; c * local_len];
                        let mut pack = vec![0.0; MatrixKernel::<f64>::pack_len(n)];
                        let mut batched_pass = |matrix: &MatrixKernel<f64>, block: &mut [f64]| {
                            for d in 0..M2L_OFFSET_COUNT {
                                let planes_d = batched.planes(LEVEL, d);
                                batched_apply(
                                    matrix,
                                    n,
                                    planes,
                                    planes_d,
                                    &block_inputs,
                                    &sources,
                                    block,
                                    &targets,
                                    &mut pack,
                                );
                            }
                        };
                        let mut expected = vec![0.0; c * local_len];
                        for d in 0..M2L_OFFSET_COUNT {
                            for (x, y) in block_inputs
                                .chunks_exact(input_len)
                                .zip(expected.chunks_exact_mut(local_len))
                            {
                                dense.m2l(LEVEL, d, x, planes, y, m);
                            }
                        }
                        let chunks = |v: &[f64]| -> Vec<Vec<f64>> {
                            v.chunks_exact(local_len).map(<[f64]>::to_vec).collect()
                        };
                        // Each mode: one pass checked against Dense, then timed.
                        let mut mode = |matrix: &MatrixKernel<f64>| {
                            block.fill(0.0);
                            batched_pass(matrix, &mut block);
                            let worst = worst_difference(&chunks(&block), &chunks(&expected));
                            let time = ns_per_pair(M2L_OFFSET_COUNT * c, || {
                                batched_pass(matrix, black_box(&mut block))
                            });
                            (time, worst)
                        };
                        (mode(&matrix), mode(&fused), *build)
                    });
                let batched_cells = match (batched_row, &dense_row) {
                    (
                        Some(((time, worst), (fused, fused_worst), build)),
                        Some((_, dense_time, _)),
                    ) => {
                        format!(
                            "{time:.0} | {fused:.0} | {:.2} | {:.2} | {:.2} | {:.2} | {build:.2} \
                             | {worst:.1e} | {fused_worst:.1e}",
                            rotation / time,
                            rotation / fused,
                            dense_time / time,
                            dense_time / fused,
                        )
                    }
                    _ => "– | – | – | – | – | – | – | – | –".into(),
                };
                let dense_mb = match dense {
                    Some(_) => format!("{:.0}", bytes as f64 / 1e6),
                    None => format!("{:.0} (skipped)", bytes as f64 / 1e6),
                };
                println!(
                    "| {p} | {k_string} | {planes} | {rotation:.0} | {dense_time} \
                     | {classes_all:.0} | {classes_z:.0} | {classes_moving:.0} \
                     | {transforms_only:.0} | {:.2} | {dense_ratio} | {dense_mb} | {:.1} \
                     | {dense_build} | {classes_build:.2} | {dense_worst} | {classes_worst:.1e} \
                     | {batched_cells} |",
                    rotation / classes_all,
                    classes.bytes() as f64 / 1e6,
                );
            }
        }
    }
}

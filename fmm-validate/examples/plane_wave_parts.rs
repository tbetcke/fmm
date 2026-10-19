//! The steps of the plane-wave M2L (Phase 6 T13, C6.5; CONVENTIONS §3.14), each timed
//! alone on synthetic data, per box: the cost of every part of the level call
//! (`nd_fmm_exec::operator`, "The plane-wave M2L") on one core and one ISA, for the
//! report and for tuning the kernels. Not registered (it times); nothing is asserted.
//!
//! On `--boxes` boxes (default 4,096, a level of 512 interior parents' worth of waves)
//! with seeded random multipoles, for each degree of `--ps` (default 3, 6, 8, 12, 18) in
//! f64 (or f32, `--precision`), the median of `--repeats` (default 5) passes of each step
//! over every box, every frame:
//!
//! - **rotate in**: the multipole into frames 1 and 2 (`rotate_multipole`), per box;
//! - **modes**: the weighted Fourier modes (`outgoing_modes`), three per box;
//! - **waves**: the waves from the modes (`WaveKernel::waves`), three per box;
//! - **merge**: the accumulators of an interior parent, every slot present
//!   (`WaveKernel::merge`, the sources' waves at random boxes), per child (248 entries per
//!   parent over the six directions, 31 per box);
//! - **gather**: every child's accumulators and its shift (`WaveKernel::gather`), per box;
//! - **sums**: the incoming sums of a child (`WaveKernel::sums`), three per box;
//! - **local**: the local in the frame (`local_from_sums`), three per box;
//! - **rotate out**: the local back from frames 1 and 2 (`add_local_from_frame`), per box.
//!
//! The sum is the model of the level call's time per box with every box interior; the
//! FMM's time per box (`scaling --kinds`, M2L over the boxes with a V row) is the check.
//!
//! ```text
//! cargo run --release -p nd-fmm-validate --example plane_wave_parts -- --isa avx512 --ps 8
//! ```

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_simd::{Isa, SimdScalar, WaveBlock, WaveKernel};
use nd_fmm_tables::PlaneWaveTables;
use nd_fmm_tables::plane_wave::Direction;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 * 2f64.powi(-53)
    }
}

const PARTS: [&str; 8] = [
    "rotate in",
    "modes",
    "waves",
    "merge",
    "gather",
    "sums",
    "local",
    "rotate out",
];

/// The median of `repeats` runs of `f`, in seconds.
fn median(repeats: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[repeats / 2]
}

/// The time per box of every part at degree `p`, in µs.
fn parts<T: SimdScalar>(p: usize, isa: Isa, boxes: usize, repeats: usize) -> [f64; 8] {
    let tables = PlaneWaveTables::<T>::build(p);
    let kernel = WaveKernel::<T>::new(isa).expect("an available ISA");
    let blocks: Vec<WaveBlock> = tables
        .blocks()
        .iter()
        .map(|b| WaveBlock {
            start: b.start,
            len: b.len,
            modes: b.modes,
            mode_start: b.mode_start,
            trig: b.trig,
        })
        .collect();
    let layout = kernel.tables(&blocks, tables.cos(), tables.sin());
    let (n, w) = ((p + 1) * (p + 1), tables.wave_len());
    let mut rng = Rng(0x9a + p as u64);
    let multipoles: Vec<T> = (0..boxes * n)
        .map(|_| T::from_f64(rng.unit() - 0.5))
        .collect();
    let mut rotated = vec![T::zero(); n];
    let mut modes = vec![T::zero(); 4 * tables.mode_count()];
    let mut work = vec![T::zero(); tables.work_len()];
    let mut waves = vec![T::zero(); boxes * 2 * w];
    let mut out = [0.0; 8];
    let per_box = |seconds: f64| 1e6 * seconds / boxes as f64;

    out[0] = per_box(median(repeats, || {
        for m in multipoles.chunks_exact(n) {
            for frame in 1..3 {
                tables.rotate_multipole(frame, m, &mut rotated);
                black_box(&rotated);
            }
        }
    }));
    out[1] = per_box(median(repeats, || {
        for m in multipoles.chunks_exact(n) {
            for _ in 0..3 {
                tables.outgoing_modes(m, &mut modes, &mut work);
                black_box(&modes);
            }
        }
    }));
    tables.outgoing_modes(&multipoles[..n], &mut modes, &mut work);
    out[2] = per_box(median(repeats, || {
        for out in waves.chunks_exact_mut(2 * w) {
            for _ in 0..3 {
                let (up, down) = out.split_at_mut(w);
                kernel.waves(&layout, &modes, up, down);
            }
        }
    }));
    // Merge and gather, parent by parent, every slot present, the sources' waves at
    // random boxes; the accumulators of the six directions.
    let parents = boxes / 8;
    let geometry = tables.geometry();
    let a = geometry.max_accumulators();
    let mut accs = vec![T::zero(); a * w];
    let sources: Vec<usize> = (0..parents * 216)
        .map(|_| (rng.unit() * boxes as f64) as usize % boxes)
        .collect();
    let (mut wave_offsets, mut shift_offsets) = (Vec::new(), Vec::new());
    let mut incoming = vec![T::zero(); w];
    let mut acc_offsets = Vec::new();
    let mut merge = 0.0;
    let mut gather = 0.0;
    for _ in 0..repeats.max(1) {
        let (mut m, mut g) = (0.0, 0.0);
        for parent in 0..parents {
            for direction in Direction::ALL {
                let half = usize::from(!direction.is_up());
                let accumulators = geometry.accumulators(direction);
                let start = Instant::now();
                for (k, accumulator) in accumulators.iter().enumerate() {
                    wave_offsets.clear();
                    shift_offsets.clear();
                    for &slot in &accumulator.slots {
                        wave_offsets.push((2 * sources[parent * 216 + slot as usize] + half) * w);
                        shift_offsets.push(
                            tables
                                .source_shift_offset(direction, slot as usize)
                                .unwrap(),
                        );
                    }
                    let acc = &mut accs[k * w..(k + 1) * w];
                    acc.fill(T::zero());
                    kernel.merge(
                        acc,
                        &waves,
                        &wave_offsets,
                        tables.source_shifts(),
                        &shift_offsets,
                    );
                }
                m += start.elapsed().as_secs_f64();
                let start = Instant::now();
                for child in 0..8 {
                    acc_offsets.clear();
                    for (k, accumulator) in accumulators.iter().enumerate() {
                        if accumulator.mask & (1 << child) != 0 {
                            acc_offsets.push(k * w);
                        }
                    }
                    kernel.gather(
                        &mut incoming,
                        &accs,
                        &acc_offsets,
                        tables.child_shift(direction, child),
                    );
                    black_box(&incoming);
                }
                g += start.elapsed().as_secs_f64();
            }
        }
        merge = if merge == 0.0 { m } else { f64::min(merge, m) };
        gather = if gather == 0.0 {
            g
        } else {
            f64::min(gather, g)
        };
    }
    out[3] = per_box(merge);
    out[4] = per_box(gather);
    let mut sums = vec![T::zero(); 4 * tables.mode_count()];
    out[5] = per_box(median(repeats, || {
        for b in 0..boxes {
            for _ in 0..3 {
                let (up, down) = waves[2 * b * w..(2 * b + 2) * w].split_at(w);
                kernel.sums(&layout, up, down, &mut sums);
                black_box(&sums);
            }
        }
    }));
    let mut frame = vec![T::zero(); n];
    out[6] = per_box(median(repeats, || {
        for _ in 0..boxes {
            for _ in 0..3 {
                tables.local_from_sums(&sums, &mut frame);
                black_box(&frame);
            }
        }
    }));
    let mut local = vec![T::zero(); n];
    out[7] = per_box(median(repeats, || {
        for _ in 0..boxes {
            for f in 1..3 {
                tables.add_local_from_frame(f, &frame, &mut local);
            }
            black_box(&local);
        }
    }));
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut isa, mut ps, mut boxes, mut repeats, mut f32_) =
        (Isa::detect(), vec![3, 6, 8, 12, 18], 4096, 5, false);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--isa" => {
                isa = Isa::all()
                    .find(|i| i.to_string() == value)
                    .unwrap_or_else(|| panic!("unknown ISA {value}"));
            }
            "--ps" => ps = value.split(',').map(|v| v.parse().unwrap()).collect(),
            "--boxes" => boxes = value.parse().unwrap(),
            "--repeats" => repeats = value.parse().unwrap(),
            "--precision" => f32_ = value == "f32",
            _ => panic!("unknown option {flag}"),
        }
    }
    assert!(isa.is_available(), "{isa} is not available here");
    println!(
        "plane-wave M2L, time per box (µs), {}, ISA {isa}, {boxes} boxes, median of {repeats}; \
         machine: {}",
        if f32_ { "f32" } else { "f64" },
        nd_fmm_validate::scaling::machine()
    );
    println!();
    println!("| p | s | H | {} | sum |", PARTS.join(" | "));
    println!(
        "| ---: | ---: | ---: | {} ---: |",
        "---: | ".repeat(PARTS.len())
    );
    for p in ps {
        let times = if f32_ {
            parts::<f32>(p, isa, boxes, repeats)
        } else {
            parts::<f64>(p, isa, boxes, repeats)
        };
        let tables = PlaneWaveTables::<f64>::build(p);
        let cells: Vec<String> = times.iter().map(|t| format!("{t:.3}")).collect();
        println!(
            "| {p} | {} | {} | {} | {:.3} |",
            tables.quadrature().nodes(),
            tables.h(),
            cells.join(" | "),
            times.iter().sum::<f64>()
        );
    }
}

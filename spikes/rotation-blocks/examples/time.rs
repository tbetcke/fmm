//! Time per block set: the Gumerov–Duraiswami y-rotation blocks of degrees 0 to p (one
//! polar angle, the compact layout) against `nd_fmm_math::rotation::blocks` plus the
//! conjugation by N, release, one thread, f64. Prints Markdown for SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-rotation-blocks --release --example time`

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_math::rotation::{blocks, blocks_len, euler_zyz};
use nd_fmm_spike_rotation_blocks::conjugated::conjugated_blocks;
use nd_fmm_spike_rotation_blocks::gd::{Scratch, compact_blocks_len, y_blocks};

/// Median seconds per call over 7 rounds, each long enough (≥ 50 ms) to time.
fn median(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    f();
    let once = start.elapsed().as_secs_f64().max(1e-7);
    let reps = ((0.05 / once).ceil() as usize).max(1);
    let mut rounds: Vec<f64> = (0..7)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() / reps as f64
        })
        .collect();
    rounds.sort_by(f64::total_cmp);
    rounds[3]
}

fn main() {
    let beta = (1.0f64 / 3.0f64.sqrt()).acos();
    let (s, x) = beta.sin_cos();
    let q = euler_zyz(0.0, beta, 0.0);
    println!(
        "| p | GD set (ms) | ns per stored value | compact set (MiB) | Racah `blocks` (ms) | `blocks` + N (ms) | full set (MiB) |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for p in [8usize, 20, 33, 50, 82, 144, 261, 500, 1000] {
        let mut out = vec![0.0; compact_blocks_len(p)];
        let mut scratch = Scratch::new(p);
        let gd = median(|| y_blocks(p, black_box(x), black_box(s), &mut out, &mut scratch));
        let mib = |values: usize| values as f64 * 8.0 / (1u64 << 20) as f64;
        let (racah, conj) = if p <= 82 {
            let mut d = vec![0.0; blocks_len(p)];
            let racah = median(|| blocks(p, black_box(&q), &mut d));
            let conj = median(|| {
                black_box(conjugated_blocks(p, black_box(&q)));
            });
            (format!("{:.3}", racah * 1e3), format!("{:.3}", conj * 1e3))
        } else {
            ("–".to_string(), "–".to_string())
        };
        println!(
            "| {p} | {:.3} | {:.2} | {:.2} | {racah} | {conj} | {:.2} |",
            gd * 1e3,
            gd * 1e9 / compact_blocks_len(p) as f64,
            mib(compact_blocks_len(p)),
            mib(blocks_len(p)),
        );
    }
}

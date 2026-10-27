//! Time per coaxial table on one thread: the recursion (triangle) against the
//! projection, for one M2M/L2L table and one M2L table (d = 2) per level, real k.
//! Prints a Markdown table for SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-coaxial --release --example time`

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_spike_coaxial::coaxial::Table;
use nd_fmm_spike_coaxial::complex::C64;
use nd_fmm_spike_coaxial::config::{degree, m2l_kt, m2m_kt, scale};
use nd_fmm_spike_coaxial::projection::{default_nodes, regular_regular, singular_regular};
use nd_fmm_spike_coaxial::{Op, start_values};

/// Best of `reps` runs, in seconds.
fn best(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn ms(s: f64) -> String {
    format!("{:.3}", s * 1e3)
}

fn main() {
    println!(
        "| k a | p_child → p | M2M recursion (ms) | M2M projection (ms) | M2L recursion (ms) | M2L projection (ms) | M2L table (MiB, halved) |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for ka in [1.0, 10.0, 30.0, 50.0, 100.0, 200.0, 300.0] {
        let p = degree(ka, 1e-10);
        let pc = degree(ka / 2.0, 1e-10);
        let reps = if p > 200 { 2 } else { 5 };
        let k = C64::new(ka, 0.0);
        let z = C64::new(m2m_kt(ka), 0.0);
        let m2m_rec = best(reps, || {
            let start = start_values(Op::Regular, z, 1.0, pc + p);
            black_box(Table::build(&start, 1.0, pc, p));
        });
        let rho = 4.0 / 3.0f64.sqrt();
        let m2m_proj = best(reps, || {
            black_box(regular_regular(k, 0.5, rho, pc, p, default_nodes(pc, p)));
        });
        let g = scale(ka);
        let z = C64::new(m2l_kt(ka, 4), 0.0);
        let mut stored = 0;
        let m2l_rec = best(reps, || {
            let start = start_values(Op::Singular, z, g, 2 * p);
            let table = Table::build(&start, g, p, p);
            stored = table.stored();
            black_box(table);
        });
        let t = m2l_kt(1.0, 4);
        let m2l_proj = best(reps, || {
            black_box(singular_regular(k, t, 1.0, g, p, p, default_nodes(p, p)));
        });
        println!(
            "| {ka} | {pc} → {p} | {} | {} | {} | {} | {:.1} |",
            ms(m2m_rec),
            ms(m2m_proj),
            ms(m2l_rec),
            ms(m2l_proj),
            (stored * 16) as f64 / (1 << 20) as f64
        );
    }
}

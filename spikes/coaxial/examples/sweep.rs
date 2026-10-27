//! The recursions against the projection over the octree's configurations, in the
//! FMM-weighted metric (src/metric.rs): one M2M/L2L table per level, the M2L tables at
//! all 15 V-list distances. Prints a Markdown table for SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-coaxial --release --example sweep [eps ...]`

use nd_fmm_spike_coaxial::coaxial::Table;
use nd_fmm_spike_coaxial::complex::C64;
use nd_fmm_spike_coaxial::config::{DISTANCES2, degree, m2l_kt, m2m_kt, scale};
use nd_fmm_spike_coaxial::metric::{compare, m2l_weights, m2m_weights};
use nd_fmm_spike_coaxial::projection::{default_nodes, regular_regular, singular_regular};
use nd_fmm_spike_coaxial::{Op, start_values};

const KA: [f64; 14] = [
    0.01, 0.1, 1.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0,
];

fn main() {
    let eps: Vec<f64> = {
        let args: Vec<f64> = std::env::args()
            .skip(1)
            .map(|s| s.parse().unwrap())
            .collect();
        if args.is_empty() { vec![1e-10] } else { args }
    };
    let kinds = [
        ("real", 1.0, 0.0),
        ("complex η=0.1", 1.0, 0.1),
        ("complex η=1", 1.0, 1.0),
        ("imaginary", 0.0, 1.0),
    ];
    println!("| k | ε | k a | p_child → p | M2M/L2L | M2L, worst of 15 (d²) |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for &e in &eps {
        for (name, re, im) in kinds {
            if e != 1e-10 && name != "real" {
                continue;
            }
            for ka in KA {
                // a = 1, so k a = |k| for real k; k a means κ a for the others.
                let k = C64::new(re * ka, im * ka);
                let p = degree(ka, e);
                let pc = degree(ka / 2.0, e);
                // M2M: child shift a/2, t along +z, (R|R) = (S|S).
                let kt = m2m_kt(ka);
                let rec = Table::build(
                    &start_values(Op::Regular, C64::new(re * kt, im * kt), 1.0, pc + p),
                    1.0,
                    pc,
                    p,
                );
                let rho = 4.0 / 3.0f64.sqrt();
                let proj = regular_regular(k, 0.5, rho, pc, p, default_nodes(pc, p));
                let (wi, wo) = m2m_weights(k, 1.0, pc, p);
                let (rr, _) = compare(
                    pc,
                    p,
                    &wi,
                    &wo,
                    |m, a, b| proj.get(m, a, b),
                    |m, a, b| rec.get(m, a, b),
                );
                // M2L at every distance.
                let g = scale(k.abs());
                let w = m2l_weights(k, 1.0, g, p);
                let mut worst = (0.0f64, 0u32);
                for d2 in DISTANCES2 {
                    let kt = m2l_kt(ka, d2);
                    let t = m2l_kt(1.0, d2);
                    let rec = Table::build(
                        &start_values(Op::Singular, C64::new(re * kt, im * kt), g, 2 * p),
                        g,
                        p,
                        p,
                    );
                    let proj = singular_regular(k, t, 1.0, g, p, p, default_nodes(p, p));
                    let (err, _) = compare(
                        p,
                        p,
                        &w,
                        &w,
                        |m, a, b| proj.get(m, a, b),
                        |m, a, b| rec.get(m, a, b),
                    );
                    if err > worst.0 {
                        worst = (err, d2);
                    }
                }
                println!(
                    "| {name} | {e:.0e} | {ka} | {pc} → {p} | {rr:.1e} | {:.1e} ({}) |",
                    worst.0, worst.1
                );
            }
        }
    }
}

//! Time per call and per value on one thread, each family with and without its
//! derivative, at the arguments and orders of the operators. Prints a Markdown table for
//! SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-radial --release --example time`

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_spike_radial::complex::C64;
use nd_fmm_spike_radial::radial::{bessel_j, hankel, modified_i, modified_k};

/// Median of 15 runs of `reps` calls, in nanoseconds per call.
fn median(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut runs: Vec<f64> = (0..15)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..reps {
                f();
            }
            start.elapsed().as_secs_f64() * 1e9 / reps as f64
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    runs[7]
}

fn main() {
    println!("| use | family | z | orders | derivative | ns per call | ns per value |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    let cases: [(&str, &str, C64, usize); 10] = [
        ("P2M, L2P", "j", C64::new(5.0, 0.0), 40),
        ("P2M, L2P", "j", C64::new(5.0, 5.0), 40),
        ("P2M, L2P", "j", C64::new(0.01, 0.0), 40),
        ("M2P, P2L", "h", C64::new(15.0, 0.0), 40),
        ("M2P, P2L", "h", C64::new(15.0, 15.0), 40),
        ("Yukawa P2M", "i", C64::new(5.0, 0.0), 40),
        ("Yukawa M2P", "k", C64::new(15.0, 0.0), 40),
        ("M2M start", "j", C64::new(325.0, 0.0), 1050),
        ("M2L start", "h", C64::new(1500.0, 0.0), 1460),
        ("Yukawa M2L start", "k", C64::new(1500.0, 0.0), 72),
    ];
    for (name, family, z, top) in cases {
        let s = 1.0f64.min(z.abs());
        for der in [false, true] {
            let mut v = vec![C64::ZERO; top + 1];
            let mut d = vec![C64::ZERO; top + 1];
            let mut vr = vec![0.0; top + 1];
            let mut dr = vec![0.0; top + 1];
            let reps = (200_000 / top).max(10);
            let ns = median(reps, || {
                let z = black_box(z);
                match (family, der) {
                    ("j", false) => bessel_j(z, s, &mut v, None),
                    ("j", true) => bessel_j(z, s, &mut v, Some(&mut d)),
                    ("h", false) => hankel(z, s, &mut v, None),
                    ("h", true) => hankel(z, s, &mut v, Some(&mut d)),
                    ("i", false) => modified_i(z.re, s, &mut vr, None),
                    ("i", true) => modified_i(z.re, s, &mut vr, Some(&mut dr)),
                    ("k", false) => modified_k(z.re, s, &mut vr, None),
                    _ => modified_k(z.re, s, &mut vr, Some(&mut dr)),
                }
                black_box((&v, &d, &vr, &dr));
            });
            println!(
                "| {name} | {family} | {} {:+} i | 0..={top} | {} | {ns:.0} | {:.1} |",
                z.re,
                z.im,
                if der { "yes" } else { "no" },
                ns / (top + 1) as f64
            );
        }
    }
}

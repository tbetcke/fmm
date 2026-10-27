//! Convergence of the projection in the quadrature order N: the weighted difference of
//! N = p + 10, 20, 40, 60 against N = 2p + 200, M2L at d = 2 and √27 and M2M, real and
//! strongly damped k (Im k = Re k). Prints a Markdown table for SPIKE_REPORT.md §2.2.
//!
//! `cargo run -p nd-fmm-spike-coaxial --release --example nodes`

use nd_fmm_spike_coaxial::complex::C64;
use nd_fmm_spike_coaxial::config::{degree, m2l_kt, scale};
use nd_fmm_spike_coaxial::metric::{compare, m2l_weights, m2m_weights};
use nd_fmm_spike_coaxial::projection::{Dense, regular_regular, singular_regular};

const EXTRA: [usize; 4] = [10, 20, 40, 60];

fn row(
    label: String,
    p_src: usize,
    p_tgt: usize,
    w: (&[f64], &[f64]),
    make: impl Fn(usize) -> Dense,
) {
    let big = make(2 * p_src.max(p_tgt) + 200);
    let mut line = format!("| {label} |");
    for extra in EXTRA {
        let t = make(p_src.max(p_tgt) + extra);
        let (err, _) = compare(
            p_src,
            p_tgt,
            w.0,
            w.1,
            |m, a, b| big.get(m, a, b),
            |m, a, b| t.get(m, a, b),
        );
        line += &format!(" {err:.1e} |");
    }
    println!("{line}");
}

fn main() {
    println!("| table | N = p + 10 | p + 20 | p + 40 | p + 60 |");
    println!("| --- | --- | --- | --- | --- |");
    for ka in [0.01, 1.0, 10.0, 30.0, 100.0, 300.0] {
        for im in [0.0, 1.0] {
            let k = C64::new(ka, im * ka);
            let p = degree(ka, 1e-10);
            let g = scale(k.abs());
            let w = m2l_weights(k, 1.0, g, p);
            for d2 in [4u32, 27] {
                let t = m2l_kt(1.0, d2);
                row(
                    format!("M2L k a = {ka}, Im k = {im} Re k, d² = {d2}, p = {p}"),
                    p,
                    p,
                    (&w, &w),
                    |n| singular_regular(k, t, 1.0, g, p, p, n),
                );
            }
            let pc = degree(ka / 2.0, 1e-10);
            let (wi, wo) = m2m_weights(k, 1.0, pc, p);
            row(
                format!("M2M k a = {ka}, Im k = {im} Re k, p = {pc} → {p}"),
                pc,
                p,
                (&wi, &wo),
                |n| regular_regular(k, 0.5, 4.0 / 3.0f64.sqrt(), pc, p, n),
            );
        }
    }
}

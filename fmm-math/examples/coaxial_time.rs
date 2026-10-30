//! Times to build one coaxial table by each generator of `nd_fmm_math::coaxial`, at the
//! degrees of proposal §3.5 (ε = 10⁻¹⁰) for real k: M2M (p_child → p_parent, the outer
//! projection above the threshold), L2L (the inner projection) and the M2L at the nearest
//! distance. Best of three, one thread. Helmholtz CH1.2c's report
//! (fmm-validate/results/helmholtz-ch1-2c.md) quotes it.
//!
//! `cargo run -p nd-fmm-math --release --example coaxial_time`

use std::time::Instant;

use nd_fmm_math::Complex;
use nd_fmm_math::coaxial::CoaxialTable;

fn degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// The best of three runs of `f`, in milliseconds.
fn best(f: impl Fn() -> CoaxialTable<f64>) -> f64 {
    (0..3)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(f());
            start.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    println!(
        "| k a | p_child → p | M2M recursion | M2M outer | L2L inner | M2L recursion | M2L projection | M2L table |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    let t = 0.5;
    let t_m2l = 4.0 / 3.0f64.sqrt();
    for ka in [1.0, 10.0, 30.0, 50.0, 100.0, 200.0, 300.0] {
        let (p, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
        let k = Complex::new(ka, 0.0);
        let s = ka.min(1.0);
        let rho = 8.0 * t / 3.0f64.sqrt();
        let n_outer = p + 40;
        let n_inner = p + 40 + (ka * t).ceil() as usize;
        let m2m_rec = best(|| CoaxialTable::regular(k * t, pc, p));
        let m2m_proj = best(|| CoaxialTable::projection_regular_outer(k, t, rho, pc, p, n_outer));
        let l2l_proj = best(|| CoaxialTable::projection_regular_inner(k, t, t, p, pc, n_inner));
        let m2l_rec = best(|| CoaxialTable::singular_regular(k * t_m2l, s, p, p));
        let m2l_proj =
            best(|| CoaxialTable::projection_singular_regular(k, t_m2l, 1.0, s, p, p, p + 40));
        let mib = CoaxialTable::singular_regular(k * t_m2l, s, p, p).stored() as f64 * 16.0
            / (1024.0 * 1024.0);
        println!(
            "| {ka} | {pc} → {p} | {m2m_rec:.2} ms | {m2m_proj:.1} ms | {l2l_proj:.1} ms | \
             {m2l_rec:.2} ms | {m2l_proj:.1} ms | {mib:.1} MiB |"
        );
    }
}

//! The range of the scaled radial values a level carries, per scale candidate
//! (SPIKE_REPORT.md §5), and what the exponential factor saves for Yukawa (§6). Prints
//! Markdown tables for SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-radial --release --example ranges`
//!
//! On a level with |k| a = ka and degree p (proposal §3.5, ε = 10⁻¹⁰), real k, the values
//! carried are ĵₙ(k a) for n ≤ p (P2M and L2P at the box's radius; the smallest is ĵ_p,
//! the largest the maximum over n) and ĥₙ(√3 k a) for n ≤ p (M2P and P2L at the nearest
//! distance). The M2L start values ĥₗ(k t), l ≤ 2p, are an intermediate of the table
//! build, whose entries carry s^{n+n′} (spike S2), so they are left out. Products of a
//! regular and a singular value do not depend on the scale; their factors do.

use nd_fmm_spike_radial::complex::C64;
use nd_fmm_spike_radial::radial::{bessel_j, hankel, modified_i, modified_k};
use nd_fmm_spike_radial::ranges::{EPS, Radius, degree};

/// The scale candidates: a power scale with one of three radii, or a per-degree scale.
#[derive(Clone, Copy)]
enum Candidate {
    /// s = min(|k| r, 1), regular values divided by sⁿ, singular multiplied.
    Power(Radius),
    /// The power scale with the circumradius and the double factorial (2n + 1)!!
    /// absorbed: σₙ = sⁿ/(2n + 1)!!.
    DoubleFactorial,
    /// σₙ the envelope max_{0 ≤ x ≤ |k| a} |jₙ(x)|.
    Envelope,
}

impl Candidate {
    fn name(self) -> String {
        match self {
            Candidate::Power(r) => format!("power, {}", r.name()),
            Candidate::DoubleFactorial => "power, a, (2n+1)!!".to_string(),
            Candidate::Envelope => "envelope".to_string(),
        }
    }
}

/// log₁₀ (2n + 1)!!.
fn log_double_factorial(n: usize) -> f64 {
    (0..=n).map(|k| ((2 * k + 1) as f64).log10()).sum()
}

/// log₁₀ σₙ of the envelope candidate, n = 0..=top, on a level with |k| a = ka: the
/// maximum of |jₙ(x)| over 200 points of (0, ka] and the end point, from the scaled
/// values ĵₙ(x; s = ka) so nothing underflows.
fn log_envelope(ka: f64, top: usize) -> Vec<f64> {
    let s = ka.min(1.0);
    let mut best = vec![f64::NEG_INFINITY; top + 1];
    let mut v = vec![C64::ZERO; top + 1];
    for i in 1..=200 {
        let x = ka * i as f64 / 200.0;
        bessel_j(C64::new(x, 0.0), s, &mut v, None);
        for (b, x) in best.iter_mut().zip(&v) {
            *b = b.max(x.abs().log10());
        }
    }
    best.iter()
        .enumerate()
        .map(|(n, b)| b + n as f64 * s.log10())
        .collect()
}

/// (min, max) of log₁₀ |value| over the values a level carries, under a candidate.
fn range(ka: f64, candidate: Candidate) -> (f64, f64) {
    let p = degree(ka, EPS);
    let sa = ka.min(1.0);
    let envelope = log_envelope(ka, p);
    // log₁₀ of the regular and singular factor of order n relative to the circumradius
    // power scale (the values are computed at sa).
    let shift = |n: usize| -> f64 {
        let n_f = n as f64;
        match candidate {
            Candidate::Power(r) => n_f * (sa / r.scale(ka)).log10(),
            Candidate::DoubleFactorial => log_double_factorial(n),
            Candidate::Envelope => n_f * sa.log10() - envelope[n],
        }
    };
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut j = vec![C64::ZERO; p + 1];
    bessel_j(C64::new(ka, 0.0), sa, &mut j, None);
    for (n, v) in j.iter().enumerate() {
        hi = hi.max(v.abs().log10() + shift(n));
    }
    lo = lo.min(j[p].abs().log10() + shift(p));
    let mut h = vec![C64::ZERO; p + 1];
    hankel(C64::new(ka * 3.0f64.sqrt(), 0.0), sa, &mut h, None);
    for (n, v) in h.iter().enumerate() {
        let l = v.abs().log10() - shift(n);
        lo = lo.min(l);
        hi = hi.max(l);
    }
    (lo, hi)
}

fn main() {
    let candidates = [
        Candidate::Power(Radius::Width),
        Candidate::Power(Radius::HalfWidth),
        Candidate::Power(Radius::Circumradius),
        Candidate::DoubleFactorial,
        Candidate::Envelope,
    ];
    println!("Range of log₁₀ |value| a level carries, real k, ε = 10⁻¹⁰ (f32: −38 to 38).\n");
    print!("| k a | p |");
    for c in candidates {
        print!(" {} |", c.name());
    }
    println!();
    println!("|{}", " --- |".repeat(candidates.len() + 2));
    for ka in [
        1e-6, 1e-3, 0.1, 0.5, 1.0, 1.7, 3.0, 10.0, 40.0, 100.0, 300.0, 650.0,
    ] {
        print!("| {ka} | {} |", degree(ka, EPS));
        for c in candidates {
            let (lo, hi) = range(ka, c);
            let mark = if lo < -37.9 || hi > 38.5 {
                " (f32 out)"
            } else {
                ""
            };
            print!(" {lo:.0} to {hi:.0}{mark} |");
        }
        println!();
    }

    println!(
        "\nYukawa: log₁₀ of the unscaled iₙ(κ a) and kₙ(√3 κ a) at n = 0 and p = 35 (f64: ±308).\n"
    );
    println!("| κ a | p | log₁₀ i₀ | log₁₀ i_p | log₁₀ k₀ | log₁₀ k_p | f64 without the factor |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for ka in [1.0, 10.0, 88.0, 100.0, 300.0, 709.0, 1000.0, 3900.0] {
        // The orders 0 and p = 35, the low-frequency degree: the exponential factor, not
        // the order, decides the range here.
        let p = degree(0.0, EPS);
        let mut i = vec![0.0; p + 1];
        let mut k = vec![0.0; p + 1];
        modified_i(ka, 1.0, &mut i, None);
        modified_k(3.0f64.sqrt() * ka, 1.0, &mut k, None);
        let e = std::f64::consts::LOG10_E;
        let li = |v: f64| v.log10() + ka * e;
        let lk = |v: f64| v.log10() - 3.0f64.sqrt() * ka * e;
        let (i0, ip, k0, kp) = (li(i[0]), li(i[p]), lk(k[0]), lk(k[p]));
        let out = [i0, ip, k0, kp].iter().any(|l| l.abs() > 307.9);
        println!(
            "| {ka} | {p} | {i0:.0} | {ip:.0} | {k0:.0} | {kp:.0} | {} |",
            if out { "out of range" } else { "in range" }
        );
    }

    // The terms (2n + 1) iₙ(κ a) kₙ(κ t) of the Yukawa series at the nearest M2L distance
    // t = 4a/√3, relative to n = 0, against Laplace's (a/t)ⁿ: the order beyond which
    // every term is below ε.
    println!("\nYukawa truncation at t = 2w = 4a/√3: the last order with a term above ε.\n");
    println!("| κ a | ε = 10⁻³ | ε = 10⁻⁶ | ε = 10⁻¹⁰ |");
    println!("| --- | --- | --- | --- |");
    let t = 4.0 / 3.0f64.sqrt();
    let last = |terms: &[f64], eps: f64| terms.iter().rposition(|x| *x > eps).unwrap_or(0);
    let top = 120;
    let laplace: Vec<f64> = (0..=top).map(|n| t.powi(-(n as i32))).collect();
    let row = |name: String, terms: &[f64]| {
        println!(
            "| {name} | {} | {} | {} |",
            last(terms, 1e-3),
            last(terms, 1e-6),
            last(terms, 1e-10)
        );
    };
    row("Laplace".to_string(), &laplace);
    for ka in [0.01f64, 1.0, 10.0, 100.0, 1000.0] {
        let s = ka.min(1.0);
        let mut i = vec![0.0; top + 1];
        let mut k = vec![0.0; top + 1];
        modified_i(ka, s, &mut i, None);
        modified_k(ka * t, s, &mut k, None);
        let terms: Vec<f64> = (0..=top)
            .map(|n| (2 * n + 1) as f64 * i[n] * k[n] / (i[0] * k[0]))
            .collect();
        row(ka.to_string(), &terms);
    }
}

//! The generator of `nd_fmm_ref::helmholtz::plane_wave` (Helmholtz CH6.1b) against this
//! spike's `generator::generate`, bit for bit: every λₖ, tₖ, wₖ, Nₖ and Fₖ at the κ and ε of
//! `tests/generator.rs` (`meets_eps` and `meets_eps_everywhere`).
//!
//! `cargo run --release -p nd-fmm-spike-pw-quadrature --example generated`
//!
//! (The spike's `C` is `num_complex::Complex64`, the type nd-fmm-math re-exports.)

use nd_fmm_ref::helmholtz::plane_wave as port;
use nd_fmm_spike_pw_quadrature::C;
use nd_fmm_spike_pw_quadrature::generator::generate;

fn main() {
    let pi = core::f64::consts::PI;
    let mut cases = vec![
        (C::new(0.0, 0.0), 1e-6),
        (C::new(5.0, 0.0), 1e-3),
        (C::new(0.0, 5.0), 1e-6),
        (C::new(10.0, 3.0), 1e-3),
        (C::new(17.529, 0.876), 1.4e-4),
    ];
    for kw in [
        C::new(0.0, 0.0),
        C::new(5.0, 0.0),
        C::new(20.0, 0.0),
        C::new(16.0 * pi, 0.0),
        C::new(0.0, 5.0),
        C::new(0.0, 30.0),
        C::new(10.0, 3.0),
        C::new(40.0, 20.0),
        C::new(10.0, 1e-3),
    ] {
        for eps in [1e-3, 1e-6, 1e-9] {
            cases.push((kw, eps));
        }
    }
    cases.extend([
        (C::new(28.524, 14.262), 1.5e-10),
        (C::new(37.949, 1.897), 1.1e-3),
        (C::new(6.533, 0.0), 1.9e-7),
    ]);
    println!("| κ | ε | nodes | Σ Nₖ | Σ (2Fₖ + 1) | kernel error | bit for bit |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    let mut all = true;
    for (kw, eps) in cases {
        let spike = generate(kw, eps);
        let ours = port::generate(kw, eps);
        let bits = |a: &[C], b: &[C]| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| {
                    x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()
                })
        };
        let same = bits(&spike.lambda, ours.lambda())
            && bits(&spike.t, ours.t())
            && bits(&spike.weight, ours.weights())
            && spike.physical == ours.alpha_counts()
            && spike.fourier == ours.fourier_counts()
            && ours.shortfall().is_none();
        all &= same;
        println!(
            "| {kw} | {eps:.1e} | {} | {} | {} | {:.2e} | {} |",
            ours.nodes(),
            ours.wave_count(),
            ours.mode_count(),
            ours.kernel_error(),
            if same { "yes" } else { "NO" }
        );
    }
    assert!(all, "the port differs from the spike");
}

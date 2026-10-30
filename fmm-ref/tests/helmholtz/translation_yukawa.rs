//! Yukawa two ways for the translations (Helmholtz CH1.3c; proposal §9 item 4): the complex
//! operators at k = iκ on the expansions of real charges against Yukawa's real operators,
//! M̂ = iⁿ M̂^Y and L̂ = i⁻ⁿ L̂^Y (CONVENTIONS §3.15, "Yukawa's real form"), RCR and direct.
//!
//! Error measure: the FMM-weighted norm of `translation_direct` (`weighted_norm`) of the
//! difference between the complex output and the mapped real one, relative to the complex
//! output's, within 1e-13. Weighted, because the projection's high-degree local
//! coefficients carry amplified rounding at low frequency (`direct`'s documentation) and the
//! two paths round differently there (the complex one by least squares, the real one by
//! îₙ alone). The number of slots that agree bit for bit is printed. RCR at κ a ∈ {0.5, 5, 50, 300}
//! (a of the parent, or of the M2L's level), direct at κ a ≤ 50 (its quadrature grows
//! with κ a); degree 12 (10 for direct), the inputs from 20 real charges.

use crate::common::{C, SplitMix64, Worst, complexify};
use crate::leaf_common::circumradius;
use crate::translation_common::{Method, OFFSETS, Translation, weighted_norm, weights};

const TOLERANCE: f64 = 1e-13;

/// iⁿ (multipole) or i⁻ⁿ (local).
fn power(n: usize, multipole: bool) -> C {
    let n = if multipole { n % 4 } else { (4 - n % 4) % 4 };
    [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ][n]
}

/// (the FMM-weighted difference relative to the complex output's, bit-equal slots, slots).
fn compare(t: Translation, method: Method, kappa_a: f64, p: usize) -> (f64, usize, usize) {
    let k = C::new(0.0, kappa_a / circumradius());
    let mut rng = SplitMix64::new(0x3c_7a);
    let sources = t.sources(&mut rng, 20);
    let charges = rng.real_charges(sources.len());
    let complex_in = t.input(k, p, &sources, &complexify(&charges));
    let real_in = t.input_yukawa(k, p, &sources, &charges);
    let complex = t.apply(method, k, p, p, &complex_in);
    let real = t.apply_yukawa(method, k, p, p, &real_in);
    let len = (p + 1) * (p + 1);
    let multipole = t.output_is_multipole();
    let mut mapped = vec![0.0; 2 * len];
    let mut equal = 0;
    for n in 0..=p {
        for i in n * n..(n + 1) * (n + 1) {
            let y = power(n, multipole) * real[i];
            (mapped[i], mapped[len + i]) = (y.re, y.im);
            if complex[i] == y.re && complex[len + i] == y.im {
                equal += 1;
            }
        }
    }
    let (_, to) = t.frames();
    let w = weights(k, &to, multipole, p);
    let diff: Vec<f64> = complex.iter().zip(&mapped).map(|(a, b)| a - b).collect();
    let size = weighted_norm(&w, &complex);
    let error = weighted_norm(&w, &diff);
    let relative = if size == 0.0 { error } else { error / size };
    (relative, equal, len)
}

#[test]
fn yukawa_two_ways() {
    let list = [
        Translation::M2m(1),
        Translation::M2m(6),
        Translation::L2l(3),
        Translation::L2l(4),
        Translation::M2l(OFFSETS[0]),
        Translation::M2l(OFFSETS[2]),
        Translation::M2l(OFFSETS[5]),
    ];
    let mut worst = Worst::default();
    for (method, cases, p) in [
        (Method::Rcr, &[0.5, 5.0, 50.0, 300.0][..], 12),
        (Method::Direct, &[0.5, 5.0, 50.0][..], 10),
    ] {
        for &kappa_a in cases {
            for t in list {
                let (e, equal, slots) = compare(t, method, kappa_a, p);
                println!(
                    "{method:?} {t:?} κa = {kappa_a}: {e:.1e}, {equal} of {slots} bit for bit"
                );
                worst.record(e, || format!("{method:?} {t:?} κa = {kappa_a}"));
            }
        }
    }
    worst.check(TOLERANCE, "Yukawa two ways");
}

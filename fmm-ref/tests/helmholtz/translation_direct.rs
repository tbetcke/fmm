//! RCR against the projection (Helmholtz CH1.3c; proposal §9 item 2, "addition theorems for
//! every translation"): `rcr` and `direct` translate the same input, of the leaf operators
//! from point charges, and their outputs are compared.
//!
//! Error measure: the FMM-weighted difference of the two outputs (as CH1.2c and spike S2).
//! An expansion of degree p is weighted per degree n by wₙ = √((2n + 1)/(4π)) |fₙ(kR)|, the
//! size of degree n's terms where an FMM evaluates it: a multipole at the nearest points
//! of its V list, R = (4 − √3) r, fₙ = 4π ik ĥₙ(kR) e^{β − Im(k)R}; a local on its sphere,
//! R = a, fₙ = ĵₙ(kR) e^{Im(k)R − β} (the leaf operators' stored forms). Its weighted norm
//! is ‖C‖_w = √(Σₙ wₙ² ‖Cₙ‖²), ‖Cₙ‖ the ℓ2 norm over the orders and both planes. The
//! difference ‖C_rcr − C_direct‖_w is compared against round-off, 1e-12 (`ROUND_OFF`):
//!
//! - relative to max(‖C_direct‖_w, ‖input‖_w), the larger of the output's size and the
//!   size of the field the input carries at its own nearest targets, which an FMM adds to
//!   the same totals: every case;
//! - relative to ‖C_direct‖_w alone: real k (no damping, the output of the input's size).
//!
//! Under strong damping an M2L's output is smaller than its input by about
//! e^{−Im(k)(|t| − 2a)} and the coaxial sum Σₙ Ê_{n′n} M̂ₙ cancels, so its error relative to
//! the output alone grows (measured 1e-10 at κ a = 30 and |t| = 3w, where the factor is
//! 1e-19); relative to the input it stays at round-off, and §5.5's pruning removes such
//! contributions. The two methods truncate the same input to the same output degrees, so
//! they agree at any degree, rectangular ones included. The norms are formed from scaled
//! terms, so tiny outputs do not underflow.
//!
//! Also: the rectangular RCR as a block of the square one, bit for bit; the real coaxial
//! step for real k; finiteness under strong damping; f32.

use nd_fmm_math::Complex;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, rcr};

use crate::common::{C, SplitMix64, Worst, complexify};
use crate::leaf_common::{KINDS, Kind, circumradius, wavenumber};
use crate::translation_common::{
    Method, OFFSETS, Translation, all_offsets, weighted_norm, weights,
};

const ROUND_OFF: f64 = 1e-12;

/// The FMM-weighted difference of RCR and direct on one translation (module
/// documentation): (‖C_rcr − C_direct‖_w, ‖C_direct‖_w, ‖input‖_w).
fn difference(t: Translation, k: C, p_in: usize, p_out: usize, seed: u64) -> [f64; 3] {
    let mut rng = SplitMix64::new(seed);
    let sources = t.sources(&mut rng, 30);
    let charges = rng.complex_charges(sources.len());
    let input = t.input(k, p_in, &sources, &charges);
    let [a, b] =
        [Method::Rcr, Method::Direct].map(|method| t.apply(method, k, p_in, p_out, &input));
    let (from, to) = t.frames();
    let w_out = weights(k, &to, t.output_is_multipole(), p_out);
    let w_in = weights(k, &from, t.input_is_multipole(), p_in);
    let diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    [
        weighted_norm(&w_out, &diff),
        weighted_norm(&w_out, &b),
        weighted_norm(&w_in, &input),
    ]
}

fn translations(children: &[usize], offsets: &[[i32; 3]]) -> Vec<Translation> {
    let mut list: Vec<Translation> = children.iter().map(|&c| Translation::M2m(c)).collect();
    list.extend(children.iter().map(|&c| Translation::L2l(7 - c)));
    list.extend(offsets.iter().map(|&o| Translation::M2l(o)));
    list
}

/// Checks every translation of `list` for k = `kind` with |k| a = `ka` (a of the parent
/// and of the M2L's level) at degrees (p_in, p_out).
fn check(list: &[Translation], kind: Kind, ka: f64, p: [usize; 2], worst: &mut Worst) {
    let k = wavenumber(kind, ka / circumradius());
    check_k(list, k, &format!("{kind:?} |k|a = {ka}"), p, worst);
}

/// As [`check`] for a given k; the relative-to-output measure is asserted for real k.
fn check_k(list: &[Translation], k: C, name: &str, p: [usize; 2], worst: &mut Worst) {
    for (n, &t) in list.iter().enumerate() {
        let [diff, output, input] = difference(t, k, p[0], p[1], 0xc13c + n as u64);
        let relative = diff / output;
        let label = || format!("{t:?} {name} p = {p:?}: relative to the output {relative:.1e}");
        worst.record(diff / output.max(input), label);
        if relative > ROUND_OFF {
            println!("damped, not asserted: {}", label());
        }
        if k.im == 0.0 {
            worst.record(relative, || format!("{}, real k", label()));
        }
    }
}

#[test]
fn rcr_against_direct() {
    let list = translations(&[0, 3, 6], &OFFSETS);
    let mut worst = Worst::default();
    for kind in KINDS {
        check(&list, kind, 0.5, [10, 10], &mut worst);
        check(&list, kind, 5.0, [12, 12], &mut worst);
    }
    worst.check(ROUND_OFF, "RCR against direct");
}

#[test]
fn rcr_against_direct_rectangular() {
    let list = translations(&[2, 5], &OFFSETS[..4]);
    let mut worst = Worst::default();
    for kind in KINDS {
        check(&list, kind, 5.0, [8, 14], &mut worst);
        check(&list, kind, 5.0, [14, 8], &mut worst);
    }
    worst.check(ROUND_OFF, "RCR against direct, rectangular");
}

#[test]
#[ignore = "slow: run with --release -- --ignored"]
fn rcr_against_direct_large() {
    let children: Vec<usize> = (0..8).collect();
    let mut worst = Worst::default();
    // Every child and the eight offsets at |k| a = 30, every kind.
    let list = translations(&children, &OFFSETS);
    for kind in KINDS {
        check(&list, kind, 30.0, [40, 40], &mut worst);
    }
    // Above the projection threshold Re(k) a = 40: real and weakly damped k.
    let list = translations(&[0, 7], &OFFSETS[..4]);
    for kind in [Kind::Real, Kind::Weak] {
        check(&list, kind, 50.0, [60, 60], &mut worst);
    }
    // Damping Im(k) a = 100: Re(k) a = 20, and Yukawa's k on the complex path. (A k with
    // arg π/4 would need Re(k) a ≈ 100 and p above it, too costly for the projection.)
    let list = translations(&[1, 6], &OFFSETS);
    let damped = C::new(20.0, 100.0) / circumradius();
    check_k(&list, damped, "k a = 20 + 100i", [60, 60], &mut worst);
    check(&list, Kind::Imaginary, 100.0, [60, 60], &mut worst);
    worst.check(ROUND_OFF, "RCR against direct, large");
}

#[test]
#[ignore = "slow: run with --release -- --ignored"]
fn rcr_against_direct_every_offset() {
    let offsets = all_offsets();
    assert_eq!(offsets.len(), 316);
    let list = translations(&[], &offsets);
    let mut worst = Worst::default();
    for kind in [Kind::Real, Kind::Complex] {
        check(&list, kind, 5.0, [12, 12], &mut worst);
    }
    worst.check(ROUND_OFF, "RCR against direct, 316 offsets");
}

/// RCR with (p_in, p_out) against the square RCR at max(p_in, p_out) on the zero-padded
/// input, its output truncated: bit for bit (the coaxial table's rectangular block and
/// the degree-wise rotation blocks are those of the square, CH1.2c and CH1.2a).
#[test]
fn rectangular_is_a_block_of_the_square() {
    let k = wavenumber(Kind::Complex, 5.0 / circumradius());
    for t in translations(&[3], &OFFSETS[..2]) {
        for (p_in, p_out) in [(6, 11), (11, 6)] {
            let p = p_in.max(p_out);
            let mut rng = SplitMix64::new(7);
            let sources = t.sources(&mut rng, 20);
            let charges = rng.complex_charges(sources.len());
            let input = t.input(k, p_in, &sources, &charges);
            let rectangular = t.apply(Method::Rcr, k, p_in, p_out, &input);
            let square = t.apply(Method::Rcr, k, p, p, &pad(&input, p_in, p));
            assert_eq!(
                rectangular,
                truncate(&square, p, p_out),
                "{t:?} {p_in} → {p_out}"
            );
        }
    }
}

/// The two planes of degree `from` padded with zeros to degree `to`.
fn pad(x: &[f64], from: usize, to: usize) -> Vec<f64> {
    let (a, b) = ((from + 1) * (from + 1), (to + 1) * (to + 1));
    let mut out = vec![0.0; 2 * b];
    out[..a].copy_from_slice(&x[..a]);
    out[b..b + a].copy_from_slice(&x[a..]);
    out
}

/// The two planes of degree `from` truncated to degree `to`.
fn truncate(x: &[f64], from: usize, to: usize) -> Vec<f64> {
    let (a, b) = ((from + 1) * (from + 1), (to + 1) * (to + 1));
    [&x[..b], &x[a..a + b]].concat()
}

/// For real k the rotations and the (S|S) = (R|R) tables are real (below the projection
/// threshold the recursion's table exactly): an input with a zero imaginary plane keeps it
/// exactly zero through RCR M2M and L2L (proposal §5.2, the real coaxial step).
#[test]
fn real_k_keeps_real_expansions_real() {
    for ka in [0.5, 5.0, 30.0] {
        let k = C::new(ka / circumradius(), 0.0);
        for t in [Translation::M2m(5), Translation::L2l(2)] {
            let p = 16;
            let mut rng = SplitMix64::new(11);
            let sources = t.sources(&mut rng, 20);
            let charges = complexify(&rng.real_charges(sources.len()));
            let mut input = t.input(k, p, &sources, &charges);
            let len = (p + 1) * (p + 1);
            input[len..].fill(0.0);
            let out = t.apply(Method::Rcr, k, p, p, &input);
            assert!(
                out[len..].iter().all(|&x| x == 0.0),
                "{t:?} |k|a = {ka}: the imaginary plane is not zero"
            );
            assert!(out[..len].iter().any(|&x| x != 0.0));
        }
    }
}

/// Im(k) a = 1000, where e^β overflows: every RCR output finite, complex k and Yukawa.
#[test]
fn strong_damping_is_finite() {
    let p = 12;
    for kind in [Kind::Complex, Kind::Imaginary] {
        let k = wavenumber(kind, 1000.0 / circumradius());
        let k = C::new(k.re, k.im) * (1000.0 / (k.im * circumradius()));
        for t in translations(&[0], &OFFSETS[..1]) {
            let mut rng = SplitMix64::new(13);
            let sources = t.sources(&mut rng, 10);
            let charges = rng.complex_charges(sources.len());
            let input = t.input(k, p, &sources, &charges);
            assert!(input.iter().all(|x| x.is_finite()));
            let out = t.apply(Method::Rcr, k, p, p, &input);
            assert!(out.iter().all(|x| x.is_finite()), "{t:?} {kind:?}");
            if matches!(kind, Kind::Imaginary) {
                let real = rng.real_charges(sources.len());
                let input = t.input_yukawa(k, p, &sources, &real);
                let out = t.apply_yukawa(Method::Rcr, k, p, p, &input);
                assert!(out.iter().all(|x| x.is_finite()), "{t:?} Yukawa");
            }
        }
    }
}

/// f32 at p = 8 and s = 1 (|k| a = 5): RCR against f64, relative ℓ2 over the
/// coefficients within 1e-5 (CONVENTIONS §3.15 reserves f32; checked at low degree only).
#[test]
fn f32_against_f64() {
    let p = 8;
    let k = wavenumber(Kind::Complex, 5.0 / circumradius());
    let k32 = Complex::new(k.re as f32, k.im as f32);
    for t in translations(&[4], &OFFSETS[..2]) {
        let mut rng = SplitMix64::new(17);
        let sources = t.sources(&mut rng, 20);
        let charges = rng.complex_charges(sources.len());
        let input = t.input(k, p, &sources, &charges);
        let exact = t.apply(Method::Rcr, k, p, p, &input);
        let (from, to) = t.frames();
        let frame = |f: Frame<f64>| Frame::new(f.centre.map(|x| x as f32), f.radius as f32);
        let (from, to) = (frame(from), frame(to));
        let (a, b) = (Level::new(k32, from.radius), Level::new(k32, to.radius));
        let input: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        let mut out = vec![0.0_f32; exact.len()];
        let mut ws = Workspace::<f32>::new(p);
        match t {
            Translation::M2m(_) => {
                let table = rcr::m2m_table(p, p, &a);
                rcr::m2m(p, p, &a, &b, &from, &to, &table, &mut ws, &input, &mut out);
            }
            Translation::L2l(_) => {
                let table = rcr::l2l_table(p, p, &b);
                rcr::l2l(p, p, &a, &b, &from, &to, &table, &mut ws, &input, &mut out);
            }
            Translation::M2l(_) => {
                let table = rcr::m2l_table(p, p, &a, &from, &to);
                rcr::m2l(p, p, &a, &from, &to, &table, &mut ws, &input, &mut out);
            }
        }
        let num: f64 = out
            .iter()
            .zip(&exact)
            .map(|(&x, &y)| (x as f64 - y).powi(2))
            .sum();
        let den: f64 = exact.iter().map(|y| y * y).sum();
        let error = (num / den).sqrt();
        println!("f32 {t:?}: {error:.1e}");
        assert!(error < 1e-5, "{t:?}: {error:.2e}");
    }
}

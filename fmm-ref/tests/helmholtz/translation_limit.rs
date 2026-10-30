//! The limit k → 0 of the translations (Helmholtz CH1.3c; proposal §3.4 and §9 item 3):
//! Laplace coefficients of real charges (CONVENTIONS §3.7), mapped to the Helmholtz stored
//! form by the diagonal map of §3.4 (`leaf_limit`), translated by the Helmholtz RCR and
//! direct operators, against Laplace's `rotation::{m2m, l2l, m2l}` mapped the same way.
//!
//! # The map
//!
//! As `leaf_limit`, per level of half-width r: M̂ₙᵐ ≈ D^M M̃ₙᵐ with
//! D^M = k̂ⁿ 3^{−n/2} √((2n + 1)/(4π)) cₘ νₘ σₘ / (2n + 1)!!, independent of r, and
//! L̂ₙᵐ ≈ D^L L̃ₙᵐ with D^L = 4π (2n − 1)!! k̂⁻ⁿ 3^{n/2} √((2n + 1)/(4π)) cₘ σₘ / (νₘ r)
//! (k̂ = k/|k|, and k̂ = 1 for Yukawa's real coefficients).
//!
//! # Error measure and bounds
//!
//! Per output degree n: ‖Ĥₙ − D Lₙ‖₂ over the orders, Ĥ the Helmholtz output on the mapped
//! input and L Laplace's output, relative to Bₙ = Σⱼ |qⱼ| ‖D Lₙ(source j)‖₂, Laplace's
//! output of each source alone. With x = |k| d, d the translation's size (a_P for M2M and
//! L2L, |t| + a for the M2L), the Taylor remainders of the radial functions and of the
//! damping factors (e^{−Im z} on the shift tables, e^{2β − Im(k)|t|} on the M2L) give:
//!
//! - every path: within x (the damping, first order in Im k);
//! - real k (no damping): the real parts within x², the imaginary parts within x at degree
//!   0 and x² above (§9: k a for the imaginary part of degree 0 only; M2M and L2L are real,
//!   imaginary parts 0).
//!
//! Each plus round-off (1e-14 for RCR; 1e-11 for the projection, whose division by
//! ĵₙ(ka) amplifies its quadrature's rounding in the high-degree local coefficients at low
//! frequency), at |k| a = 10⁻², 10⁻⁴ and 10⁻⁶ (a the circumradius of the
//! parent, or of the M2L's level), so each error is shown to fall with its power of k.

use nd_fmm_ref::{Frame, Workspace as LaplaceWorkspace, leaf as laplace, rotation};

use crate::common::{C, SplitMix64, Worst};
use crate::leaf_common::{KINDS, Kind, circumradius, wavenumber};
use crate::translation_common::{Method, OFFSETS, Translation};

const P: usize = 8;
/// Round-off per method: RCR's, and the projection's, whose division by ĵₙ(ka) amplifies
/// the quadrature's rounding in the high-degree local coefficients at low frequency
/// (`direct`'s documentation).
fn round_off(method: Method) -> f64 {
    match method {
        Method::Rcr => 1e-14,
        Method::Direct => 1e-11,
    }
}
const MODULI: [f64; 3] = [1e-2, 1e-4, 1e-6];

/// The complex operators for a kind of k, or Yukawa's real ones.
#[derive(Clone, Copy, Debug)]
enum Path {
    Complex(Kind),
    Yukawa,
}

const PATHS: [Path; 5] = [
    Path::Complex(KINDS[0]),
    Path::Complex(KINDS[1]),
    Path::Complex(KINDS[2]),
    Path::Complex(KINDS[3]),
    Path::Yukawa,
];

impl Path {
    fn k(self, ka: f64) -> C {
        let modulus = ka / circumradius();
        match self {
            Self::Complex(kind) => wavenumber(kind, modulus),
            Self::Yukawa => C::new(0.0, modulus),
        }
    }

    fn phase(self, k: C) -> C {
        match self {
            Self::Complex(_) => k / k.norm(),
            Self::Yukawa => C::new(1.0, 0.0),
        }
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// n!! for odd n ≥ −1.
fn double_factorial(n: isize) -> f64 {
    (1..=n).rev().step_by(2).map(|i| i as f64).product()
}

fn slots(n: usize) -> impl Iterator<Item = (isize, usize)> {
    let n = n as isize;
    (-n..=n).map(move |m| (m, (n * n + n + m) as usize))
}

/// D^M or D^L of the module documentation for a level of half-width `half`.
fn map(phase: C, multipole: bool, half: f64, n: usize, m: isize) -> C {
    let a = m.unsigned_abs();
    let nu = (factorial(n + a) * factorial(n - a)).sqrt();
    let c = if m == 0 { 1.0 } else { 2.0_f64.sqrt() };
    let sigma = if m < 0 { -1.0 } else { 1.0 };
    let norm = ((2 * n + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
    if multipole {
        phase.powu(n as u32)
            * (3.0_f64.powf(-(n as f64) / 2.0) * c * nu * sigma * norm
                / double_factorial(2 * n as isize + 1))
    } else {
        phase.powi(-(n as i32))
            * (4.0
                * std::f64::consts::PI
                * double_factorial(2 * n as isize - 1)
                * 3.0_f64.powf(n as f64 / 2.0)
                * c
                * sigma
                * norm
                / (nu * half))
    }
}

/// Laplace's input (P2M or P2L at the input frame) and its translation.
fn laplace(t: Translation, sources: &[[f64; 3]], charges: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (from, to) = t.frames();
    let mut ws = LaplaceWorkspace::new(P);
    let len = (P + 1) * (P + 1);
    let mut input = vec![0.0; len];
    if t.input_is_multipole() {
        laplace::p2m(P, &from, sources, charges, &mut ws, &mut input);
    } else {
        laplace::p2l(P, &from, sources, charges, &mut ws, &mut input);
    }
    let mut output = vec![0.0; len];
    match t {
        Translation::M2m(_) => rotation::m2m(P, &from, &to, &mut ws, &input, &mut output),
        Translation::L2l(_) => rotation::l2l(P, &from, &to, &mut ws, &input, &mut output),
        Translation::M2l(_) => rotation::m2l(P, &from, &to, &mut ws, &input, &mut output),
    }
    (input, output)
}

/// Laplace coefficients mapped to complex Helmholtz coefficients on `frame`.
fn mapped(phase: C, multipole: bool, frame: &Frame<f64>, x: &[f64]) -> Vec<C> {
    let mut out = vec![C::new(0.0, 0.0); x.len()];
    for n in 0..=P {
        for (m, i) in slots(n) {
            out[i] = map(phase, multipole, frame.radius, n, m) * x[i];
        }
    }
    out
}

/// The Helmholtz translation of complex coefficients on a path.
fn helmholtz(path: Path, method: Method, t: Translation, k: C, input: &[C]) -> Vec<C> {
    match path {
        Path::Complex(_) => {
            let mut stored: Vec<f64> = input.iter().map(|c| c.re).collect();
            stored.extend(input.iter().map(|c| c.im));
            let out = t.apply(method, k, P, P, &stored);
            let len = input.len();
            (0..len).map(|i| C::new(out[i], out[len + i])).collect()
        }
        Path::Yukawa => {
            let stored: Vec<f64> = input
                .iter()
                .map(|c| {
                    assert_eq!(c.im, 0.0);
                    c.re
                })
                .collect();
            let out = t.apply_yukawa(method, k, P, P, &stored);
            out.into_iter().map(|x| C::new(x, 0.0)).collect()
        }
    }
}

/// Per output degree: (‖Ĥₙ − D Lₙ‖, ‖Re‖, ‖Im‖) / Bₙ.
fn errors(path: Path, method: Method, t: Translation, ka: f64) -> Vec<[f64; 3]> {
    let k = path.k(ka);
    let phase = path.phase(k);
    let (from, to) = t.frames();
    let mut rng = SplitMix64::new(0x3c_1117);
    let sources = t.sources(&mut rng, 20);
    let charges = rng.real_charges(sources.len());
    let (input, output) = laplace(t, &sources, &charges);
    let out_multipole = t.output_is_multipole();
    let expected = mapped(phase, out_multipole, &to, &output);
    let result = helmholtz(
        path,
        method,
        t,
        k,
        &mapped(phase, t.input_is_multipole(), &from, &input),
    );
    let mut magnitude = [0.0; P + 1];
    for (&y, &q) in sources.iter().zip(&charges) {
        let (_, single) = laplace(t, &[y], &[1.0]);
        let single = mapped(phase, out_multipole, &to, &single);
        for (n, b) in magnitude.iter_mut().enumerate() {
            let norm: f64 = slots(n).map(|(_, i)| single[i].norm_sqr()).sum();
            *b += q.abs() * norm.sqrt();
        }
    }
    (0..=P)
        .map(|n| {
            let (mut all, mut re, mut im) = (0.0, 0.0, 0.0);
            for (_, i) in slots(n) {
                let d = result[i] - expected[i];
                all += d.norm_sqr();
                re += d.re * d.re;
                im += d.im * d.im;
            }
            [all, re, im].map(|x| x.sqrt() / magnitude[n])
        })
        .collect()
}

/// The translation's size d of the module documentation, in units of a (the circumradius
/// of the parent, or of the M2L's level).
fn size(t: Translation) -> f64 {
    match t {
        Translation::M2m(_) | Translation::L2l(_) => 1.0,
        Translation::M2l(offset) => {
            let w = 2.0 * crate::leaf_common::HALF;
            let t: f64 = offset.iter().map(|&o| (o as f64 * w).powi(2)).sum();
            t.sqrt() / circumradius() + 1.0
        }
    }
}

#[test]
fn translations_in_the_limit() {
    let list = [
        Translation::M2m(2),
        Translation::M2m(7),
        Translation::L2l(0),
        Translation::L2l(5),
        Translation::M2l(OFFSETS[0]),
        Translation::M2l(OFFSETS[1]),
        Translation::M2l(OFFSETS[4]),
    ];
    for method in [Method::Rcr, Method::Direct] {
        let (mut all, mut real, mut imaginary) =
            (Worst::default(), Worst::default(), Worst::default());
        let floor = round_off(method);
        for t in list {
            for path in PATHS {
                for ka in MODULI {
                    let x = ka * size(t);
                    for (n, [e, re, im]) in errors(path, method, t, ka).into_iter().enumerate() {
                        let label = || format!("{method:?} {t:?} {path:?} |k|a = {ka:e} n = {n}");
                        all.record(e / (x + floor), label);
                        if matches!(path, Path::Complex(Kind::Real)) {
                            real.record(re / (x * x + floor), label);
                            let bound = if n == 0 { x } else { x * x };
                            imaginary.record(im / (bound + floor), label);
                        }
                    }
                }
            }
        }
        all.check(1.0, &format!("{method:?}, error / |k|d"));
        real.check(1.0, &format!("{method:?}, real k, real parts / (|k|d)²"));
        imaginary.check(
            1.0,
            &format!("{method:?}, real k, imaginary parts / (|k|d at n = 0, (|k|d)² above)"),
        );
    }
}

//! Yukawa two ways for the leaf operators (Helmholtz CH1.3b; proposal §9 item 4,
//! CONVENTIONS §3.15 "Yukawa's real form"): the complex operators at k = iκ against
//! Yukawa's real ones, M̂ = iⁿ M̂ᵞ and L̂ = i⁻ⁿ L̂ᵞ, and their potentials and gradients.
//!
//! The two paths use different functions of `nd_fmm_math::radial` (ĵ and ĥ of complex
//! z = iκρ against î and k̂ of real x = κρ). At z = iκρ the complex recursions repeat the
//! real ones up to exact factors of i: the coefficients and potentials agree to the last
//! bit on the M3 Max and the gradients within 4e-17 (measured; the comparison is
//! sensitive: the wrong power of i gives errors of order 1). The test asserts round-off
//! only. For real charges the complex path's potentials and gradients are real to
//! round-off.
//!
//! Error measures: per degree, ‖Ĉ − iⁿ Ĉᵞ‖₂ over the orders relative to the magnitude
//! Σⱼ |qⱼ| ‖Ĉᵞₙ(source j)‖₂ of the real path, one source at a time; the potentials and
//! gradients in the relative ℓ2 error of proposal §9.1 over the targets. Bound 1e-13. Complex charges are two real vectors on the
//! Yukawa path.

use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};

use crate::common::{C, SplitMix64, Worst, rel_l2, rel_l2_gradient};
use crate::leaf_common::{HALF, circumradius, frame, inside, shell};

const P: usize = 24;
const TOL: f64 = 1e-13;

fn i_pow(n: isize) -> C {
    [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ][n.rem_euclid(4) as usize]
}

/// The Euclidean norm, by `hypot`: the values reach 1e-196 at κ a = 300, where squares
/// underflow.
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::hypot)
}

/// The degree n of each slot.
fn degree(i: usize) -> usize {
    (i as f64).sqrt() as usize
}

struct Expansions {
    complex: Vec<C>,
    /// The real path on Re q and on Im q.
    real: [Vec<f64>; 2],
    /// The real path on |q| one source at a time, per degree: Σⱼ |qⱼ| ‖Ĉᵞₙ(j)‖.
    magnitude: Vec<f64>,
}

fn expansions(
    level: &Level<f64>,
    multipole: bool,
    sources: &[[f64; 3]],
    charges: &[C],
) -> Expansions {
    let len = (P + 1) * (P + 1);
    let mut ws = Workspace::new(P);
    let frame = frame();
    let mut complex = vec![0.0; 2 * len];
    let complex_form = if multipole { leaf::p2m } else { leaf::p2l };
    let real_form = if multipole {
        leaf::p2m_yukawa
    } else {
        leaf::p2l_yukawa
    };
    complex_form(P, level, &frame, sources, charges, &mut ws, &mut complex);
    let real = [|q: &C| q.re, |q: &C| q.im].map(|part| {
        let q: Vec<f64> = charges.iter().map(part).collect();
        let mut out = vec![0.0; len];
        real_form(P, level, &frame, sources, &q, &mut ws, &mut out);
        out
    });
    let mut magnitude = vec![0.0; P + 1];
    for (&y, q) in sources.iter().zip(charges) {
        let mut single = vec![0.0; len];
        real_form(P, level, &frame, &[y], &[1.0], &mut ws, &mut single);
        for (n, b) in magnitude.iter_mut().enumerate() {
            *b += q.norm() * norm(single[n * n..(n + 1) * (n + 1)].iter().copied());
        }
    }
    Expansions {
        complex: (0..len)
            .map(|i| C::new(complex[i], complex[len + i]))
            .collect(),
        real,
        magnitude,
    }
}

#[test]
fn coefficients_and_evaluations_agree() {
    let a = circumradius();
    let (mut coefficients, mut potentials, mut gradients) =
        (Worst::default(), Worst::default(), Worst::default());
    for kappa_a in [0.5, 5.0, 50.0, 300.0] {
        let level = Level::new(C::new(0.0, kappa_a / a), HALF);
        let mut rng = SplitMix64::new(0x3b_4a11 + kappa_a as u64);
        let inner = inside(&mut rng, 30, false);
        let outer = shell(&mut rng, 30, 2.5 * a);
        let charges = rng.complex_charges(30 + 8);
        let targets_out = shell(&mut rng, 20, 2.5 * a);
        let targets_in = inside(&mut rng, 20, true);
        for (multipole, sources, targets) in
            [(true, &inner, &targets_out), (false, &outer, &targets_in)]
        {
            let name = if multipole { "P2M" } else { "P2L" };
            let q = &charges[..sources.len()];
            let e = expansions(&level, multipole, sources, q);
            // M̂ = iⁿ M̂ᵞ, L̂ = i⁻ⁿ L̂ᵞ.
            let sign = if multipole { 1 } else { -1 };
            let len = (P + 1) * (P + 1);
            let real_as_complex: Vec<C> = (0..len)
                .map(|i| {
                    let n = degree(i) as isize;
                    i_pow(sign * n) * C::new(e.real[0][i], e.real[1][i])
                })
                .collect();
            for n in 0..=P {
                let slots = n * n..(n + 1) * (n + 1);
                let diff = norm(slots.map(|i| (e.complex[i] - real_as_complex[i]).norm()));
                let error = diff / e.magnitude[n];
                coefficients.record(error, || format!("{name} κa = {kappa_a} n = {n}"));
            }
            // Evaluate both.
            let mut ws = Workspace::new(P);
            let frame = frame();
            let mut stored: Vec<f64> = e.complex.iter().map(|c| c.re).collect();
            stored.extend(e.complex.iter().map(|c| c.im));
            let mut phi = vec![C::new(0.0, 0.0); targets.len()];
            let mut grad = vec![[C::new(0.0, 0.0); 3]; targets.len()];
            let complex_eval = if multipole { leaf::m2p } else { leaf::l2p };
            let real_eval = if multipole {
                leaf::m2p_yukawa
            } else {
                leaf::l2p_yukawa
            };
            complex_eval(
                P,
                &level,
                &frame,
                &stored,
                targets,
                &mut ws,
                &mut phi,
                Some(&mut grad),
            );
            let mut real_phi = [vec![0.0; targets.len()], vec![0.0; targets.len()]];
            let mut real_grad = [vec![[0.0; 3]; targets.len()], vec![[0.0; 3]; targets.len()]];
            for part in 0..2 {
                real_eval(
                    P,
                    &level,
                    &frame,
                    &e.real[part],
                    targets,
                    &mut ws,
                    &mut real_phi[part],
                    Some(&mut real_grad[part]),
                );
            }
            let expected: Vec<C> = (0..targets.len())
                .map(|t| C::new(real_phi[0][t], real_phi[1][t]))
                .collect();
            let expected_grad: Vec<[C; 3]> = (0..targets.len())
                .map(|t| core::array::from_fn(|c| C::new(real_grad[0][t][c], real_grad[1][t][c])))
                .collect();
            // At κ a = 300 the values are near 1e-196 and their squares underflow: the norms
            // are taken after scaling by the largest expected value.
            let scale = 1.0 / expected.iter().map(|c| c.norm()).fold(0.0, f64::max);
            let scale_grad = 1.0
                / expected_grad
                    .iter()
                    .flatten()
                    .map(|c| c.norm())
                    .fold(0.0, f64::max);
            let scaled = |v: &[C], s: f64| v.iter().map(|c| c * s).collect::<Vec<_>>();
            let scaled_grad =
                |v: &[[C; 3]], s: f64| v.iter().map(|g| g.map(|c| c * s)).collect::<Vec<_>>();
            let ep = rel_l2(&scaled(&phi, scale), &scaled(&expected, scale));
            let eg = rel_l2_gradient(
                &scaled_grad(&grad, scale_grad),
                &scaled_grad(&expected_grad, scale_grad),
            );
            potentials.record(ep, || format!("{name} κa = {kappa_a}"));
            gradients.record(eg, || format!("{name} κa = {kappa_a}"));
        }
    }
    coefficients.check(TOL, "coefficients, complex against real");
    potentials.check(TOL, "potentials, complex against real");
    gradients.check(TOL, "gradients, complex against real");
}

/// Real charges on the complex path at k = iκ: the potentials and gradients of
/// P2M → M2P and P2L → L2P are real (CONVENTIONS §3.15, "Yukawa's real form"). Error
/// measure: ‖Im‖₂ / ‖·‖₂ over the targets (after scaling, as above); bound 1e-13.
#[test]
fn real_charges_give_real_values() {
    let a = circumradius();
    let mut worst = Worst::default();
    for kappa_a in [0.5, 5.0, 50.0, 300.0] {
        let k = C::new(0.0, kappa_a / a);
        let mut rng = SplitMix64::new(0x3b_4e01 + kappa_a as u64);
        let inner = inside(&mut rng, 30, false);
        let outer = shell(&mut rng, 30, 2.5 * a);
        let charges: Vec<C> = rng
            .real_charges(38)
            .iter()
            .map(|&q| C::new(q, 0.0))
            .collect();
        let targets_out = shell(&mut rng, 20, 2.5 * a);
        let targets_in = inside(&mut rng, 20, true);
        for (multipole, sources, targets) in
            [(true, &inner, &targets_out), (false, &outer, &targets_in)]
        {
            let q = &charges[..sources.len()];
            let field = if multipole {
                crate::leaf_common::multipole_series(P, k, sources, q, targets)
            } else {
                crate::leaf_common::local_series(P, k, sources, q, targets)
            };
            let values: Vec<C> = field
                .potential
                .iter()
                .chain(field.gradient.iter().flatten())
                .copied()
                .collect();
            let scale = 1.0 / values.iter().map(|c| c.norm()).fold(0.0, f64::max);
            let imaginary = norm(values.iter().map(|c| c.im * scale));
            let total = norm(values.iter().map(|c| c.norm() * scale));
            let name = if multipole { "P2M→M2P" } else { "P2L→L2P" };
            worst.record(imaginary / total, || format!("{name} κa = {kappa_a}"));
        }
    }
    worst.check(TOL, "imaginary parts for real charges");
}

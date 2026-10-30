//! The limit k → 0 of the leaf operators (Helmholtz CH1.3b; proposal §3.4 and §9 item 3):
//! on the fixed box of `leaf_common`, the stored Helmholtz coefficients against Laplace's
//! (CONVENTIONS §3.7) under the diagonal map of §3.4, and M2P and L2P on mapped Laplace
//! coefficients against Laplace's `m2p` and `l2p`.
//!
//! # The map
//!
//! With k̂ = k/|k|, cₘ = √2 for m ≠ 0 (1 for m = 0), νₘ = √((n + |m|)!(n − |m|)!) and
//! σₘ = −1 in the slots m < 0 (Laplace stores conj Rₙᵐ, §3.7), the relation
//! rⁿ Yₙᵐ = √((2n + 1)/(4π)) (N Rₙ)ₘ of §3.15, jₙ(z) ≈ zⁿ/(2n + 1)!!,
//! hₙ(z) ≈ −i (2n − 1)!!/zⁿ⁺¹, s = |k| a and a = √3 r give, for real charges,
//!
//! ```text
//! M̂ₙᵐ ≈ D^M M̃ₙᵐ,  D^M = k̂ⁿ 3^{−n/2} √((2n + 1)/(4π)) cₘ νₘ σₘ / (2n + 1)!!
//! L̂ₙᵐ ≈ D^L L̃ₙᵐ,  D^L = 4π (2n − 1)!! k̂⁻ⁿ 3^{n/2} √((2n + 1)/(4π)) cₘ σₘ / (νₘ r)
//! ```
//!
//! For Yukawa's real path (M̂^Y = i⁻ⁿ M̂, L̂^Y = iⁿ L̂) the same with k̂ = 1.
//!
//! # Error measures and bounds
//!
//! Coefficients per degree n: ‖Ĉₙ − D Ĉ_Lₙ‖₂ over the orders, relative to
//! Bₙ = Σⱼ |qⱼ| ‖D C_Lₙ(source j)‖₂, the same map applied to each source alone. With
//! z = kρ and Im z ≥ 0, the Taylor remainders give:
//!
//! - P2M, undamped (e^β M̂): jₙ(z)(2n + 1)!!/zⁿ − 1 = −z²/(2(2n + 3)) + …, so within
//!   1.1 (|k| a)²/6; damped (M̂), with |e^{−β} − 1| ≤ β ≤ |k| a, within |k| a + (|k| a)²;
//! - P2L, undamped (e^{−β} L̂): degree 0 is 4π q e^{ikρ}/ρ, within |k| ρ_max; degrees n ≥ 1
//!   have hₙ zⁿ⁺¹/(−i (2n − 1)!!) − 1 = z²/(2(2n − 1)) + O(z³), within (|k| ρ_max)²; for
//!   real k degree 0's real part (cos kρ − 1) within (k ρ_max)²/2 (§9: (k a)² for real
//!   parts, k a for the imaginary part of degree 0).
//!
//! Evaluations per target, relative to the Laplace term magnitudes
//! Σⱼ |qⱼ| / dⱼ (potential) and 2 Σⱼ |qⱼ| / dⱼ² (gradient), with dⱼ = |v| − ρⱼ for M2P and
//! ρⱼ − |v| for L2P, which bound Σₙ of the degree-n terms and of their gradients:
//!
//! - M2P (|v| ≤ v_max = 2.5 a): each degree's factor e^β hₙ zⁿ⁺¹/(−i (2n − 1)!!) differs
//!   from 1 by at most t = |k| (a + v_max) + (|k| v_max)²; potential within t, gradient
//!   within 2t (the factor's radial derivative is of the same order); real k: the real
//!   parts within (k v_max)² and 2 (k v_max)²;
//! - L2P (|v| ≤ a): the factor e^{−β} jₙ(z)(2n + 1)!!/zⁿ differs from 1 by at most
//!   t = |k| a + (|k| a)²/6; real k: within (k a)²/6 · 1.1, and the imaginary parts
//!   at round-off.
//!
//! Each plus 1e-14 for round-off, at |k| a = 10⁻², 10⁻⁴, 10⁻⁶: each bound falls with its
//! power of k, so the errors are shown to fall with it.

use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};
use nd_fmm_ref::{Workspace as LaplaceWorkspace, leaf as laplace};

use crate::common::{C, SplitMix64, Worst, complexify, distance};
use crate::leaf_common::{
    CENTRE, HALF, KINDS, Kind, circumradius, frame, inside, shell, wavenumber,
};

const P: usize = 8;
const ROUND_OFF: f64 = 1e-14;
const MODULI: [f64; 3] = [1e-2, 1e-4, 1e-6];

/// The paths: the complex operators for a kind of k, or Yukawa's real ones.
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

    /// k̂ of the map: k/|k|, or 1 for Yukawa's real coefficients.
    fn phase(self, k: C) -> C {
        match self {
            Self::Complex(_) => k / k.norm(),
            Self::Yukawa => C::new(1.0, 0.0),
        }
    }

    fn is_real(self) -> bool {
        matches!(self, Self::Complex(Kind::Real))
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

/// n!! for odd n ≥ −1, with (−1)!! = 1.
fn double_factorial(n: isize) -> f64 {
    (1..=n).rev().step_by(2).map(|i| i as f64).product()
}

fn slots(n: usize) -> impl Iterator<Item = (isize, usize)> {
    let n = n as isize;
    (-n..=n).map(move |m| (m, (n * n + n + m) as usize))
}

/// cₘ νₘ σₘ √((2n + 1)/(4π)) and the 1/νₘ form, shared by both maps.
fn common(n: usize, m: isize) -> (f64, f64) {
    let a = m.unsigned_abs();
    let nu = (factorial(n + a) * factorial(n - a)).sqrt();
    let c = if m == 0 { 1.0 } else { 2.0_f64.sqrt() };
    let sigma = if m < 0 { -1.0 } else { 1.0 };
    let norm = ((2 * n + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt();
    (c * nu * sigma * norm, c * sigma * norm / nu)
}

fn map_multipole(phase: C, n: usize, m: isize) -> C {
    let (factor, _) = common(n, m);
    phase.powu(n as u32)
        * (3.0_f64.powf(-(n as f64) / 2.0) * factor / double_factorial(2 * n as isize + 1))
}

fn map_local(phase: C, n: usize, m: isize) -> C {
    let (_, factor) = common(n, m);
    let scale = 4.0
        * std::f64::consts::PI
        * double_factorial(2 * n as isize - 1)
        * 3.0_f64.powf(n as f64 / 2.0)
        * factor
        / HALF;
    phase.powi(-(n as i32)) * scale
}

/// The coefficients of an expansion as complex numbers per slot.
fn coefficients(path: Path, stored: &[f64]) -> Vec<C> {
    let len = (P + 1) * (P + 1);
    match path {
        Path::Complex(_) => (0..len)
            .map(|i| C::new(stored[i], stored[len + i]))
            .collect(),
        Path::Yukawa => stored.iter().map(|&x| C::new(x, 0.0)).collect(),
    }
}

/// The Helmholtz P2M or P2L of real charges on a path.
fn form(
    path: Path,
    level: &Level<f64>,
    multipole: bool,
    sources: &[[f64; 3]],
    charges: &[f64],
) -> Vec<C> {
    let mut ws = Workspace::new(P);
    let len = (P + 1) * (P + 1);
    let frame = frame();
    match path {
        Path::Complex(_) => {
            let mut out = vec![0.0; 2 * len];
            let q = complexify(charges);
            if multipole {
                leaf::p2m(P, level, &frame, sources, &q, &mut ws, &mut out);
            } else {
                leaf::p2l(P, level, &frame, sources, &q, &mut ws, &mut out);
            }
            coefficients(path, &out)
        }
        Path::Yukawa => {
            let mut out = vec![0.0; len];
            if multipole {
                leaf::p2m_yukawa(P, level, &frame, sources, charges, &mut ws, &mut out);
            } else {
                leaf::p2l_yukawa(P, level, &frame, sources, charges, &mut ws, &mut out);
            }
            coefficients(path, &out)
        }
    }
}

/// Laplace's P2M or P2L.
fn laplace_form(multipole: bool, sources: &[[f64; 3]], charges: &[f64]) -> Vec<f64> {
    let mut ws = LaplaceWorkspace::new(P);
    let mut out = vec![0.0; (P + 1) * (P + 1)];
    if multipole {
        laplace::p2m(P, &frame(), sources, charges, &mut ws, &mut out);
    } else {
        laplace::p2l(P, &frame(), sources, charges, &mut ws, &mut out);
    }
    out
}

struct Points {
    inside: Vec<[f64; 3]>,
    outside: Vec<[f64; 3]>,
    charges_inside: Vec<f64>,
    charges_outside: Vec<f64>,
    targets_inside: Vec<[f64; 3]>,
    targets_outside: Vec<[f64; 3]>,
}

fn points() -> Points {
    let a = circumradius();
    let mut rng = SplitMix64::new(0x3b_1117);
    let sources_inside = inside(&mut rng, 30, false);
    let charges_inside = rng.real_charges(sources_inside.len());
    let outside = shell(&mut rng, 30, 2.5 * a);
    let charges_outside = rng.real_charges(outside.len());
    Points {
        targets_inside: inside(&mut rng, 20, true),
        targets_outside: shell(&mut rng, 20, 2.5 * a),
        inside: sources_inside,
        outside,
        charges_inside,
        charges_outside,
    }
}

/// Per degree, ‖h·undamp − D l‖ / Bₙ, with Bₙ from the sources one at a time.
fn coefficient_errors(
    path: Path,
    k: C,
    multipole: bool,
    sources: &[[f64; 3]],
    charges: &[f64],
    undamp: f64,
) -> Vec<(f64, C, f64)> {
    let level = Level::new(k, HALF);
    let phase = path.phase(k);
    let map = |n, m| {
        if multipole {
            map_multipole(phase, n, m)
        } else {
            map_local(phase, n, m)
        }
    };
    let helmholtz = form(path, &level, multipole, sources, charges);
    let laplace_all = laplace_form(multipole, sources, charges);
    let mut magnitude = [0.0; P + 1];
    for (&y, &q) in sources.iter().zip(charges) {
        let single = laplace_form(multipole, &[y], &[1.0]);
        for (n, b) in magnitude.iter_mut().enumerate() {
            let norm: f64 = slots(n)
                .map(|(m, i)| (map(n, m) * single[i]).norm_sqr())
                .sum();
            *b += q.abs() * norm.sqrt();
        }
    }
    (0..=P)
        .map(|n| {
            let mut diff = 0.0;
            let mut degree0 = C::new(0.0, 0.0);
            for (m, i) in slots(n) {
                let d = helmholtz[i] * undamp - map(n, m) * laplace_all[i];
                diff += d.norm_sqr();
                if n == 0 {
                    degree0 = d / map(n, m);
                }
            }
            (diff.sqrt() / magnitude[n], degree0, magnitude[n])
        })
        .collect()
}

#[test]
fn p2m_coefficients() {
    let pts = points();
    let a = circumradius();
    let (mut undamped, mut damped) = (Worst::default(), Worst::default());
    for path in PATHS {
        for ka in MODULI {
            let k = path.k(ka);
            let beta = k.im * a;
            let ratio = |e: f64, bound: f64| e / (bound + ROUND_OFF);
            for (n, (e, _, _)) in
                coefficient_errors(path, k, true, &pts.inside, &pts.charges_inside, beta.exp())
                    .into_iter()
                    .enumerate()
            {
                undamped.record(ratio(e, 1.1 * ka * ka / 6.0), || {
                    format!("{path:?} |k|a = {ka:e} n = {n}: {e:.1e}")
                });
            }
            for (n, (e, _, _)) in
                coefficient_errors(path, k, true, &pts.inside, &pts.charges_inside, 1.0)
                    .into_iter()
                    .enumerate()
            {
                damped.record(ratio(e, ka + ka * ka), || {
                    format!("{path:?} |k|a = {ka:e} n = {n}: {e:.1e}")
                });
            }
        }
    }
    undamped.check(1.0, "P2M undamped, error / bound");
    damped.check(1.0, "P2M damped, error / bound");
}

#[test]
fn p2l_coefficients() {
    let pts = points();
    let a = circumradius();
    let rho_max = pts
        .outside
        .iter()
        .map(|&y| distance(y, CENTRE))
        .fold(0.0, f64::max);
    let (mut degree0, mut higher, mut real0) =
        (Worst::default(), Worst::default(), Worst::default());
    for path in PATHS {
        for ka in MODULI {
            let k = path.k(ka);
            let x = k.norm() * rho_max;
            let undamp = (-k.im * a).exp();
            let errors =
                coefficient_errors(path, k, false, &pts.outside, &pts.charges_outside, undamp);
            for (n, (e, d0, b0)) in errors.into_iter().enumerate() {
                let label = || format!("{path:?} |k|a = {ka:e} n = {n}: {e:.1e}");
                if n == 0 {
                    degree0.record(e / (x + ROUND_OFF), label);
                    if path.is_real() {
                        // The real part of degree 0, relative to the same magnitude.
                        let map0 = map_local(C::new(1.0, 0.0), 0, 0).norm();
                        let re = d0.re.abs() * map0 / b0;
                        real0.record(re / (x * x / 2.0 + ROUND_OFF), || {
                            format!("|k|a = {ka:e}: {re:.1e}")
                        });
                    }
                } else {
                    higher.record(e / (x * x + ROUND_OFF), label);
                }
            }
        }
    }
    degree0.check(1.0, "P2L degree 0, error / bound");
    higher.check(1.0, "P2L degrees ≥ 1, error / bound");
    real0.check(1.0, "P2L real k, degree 0 real part, error / bound");
}

/// Evaluates mapped Laplace coefficients with the Helmholtz M2P or L2P on a path.
fn evaluate(
    path: Path,
    k: C,
    multipole: bool,
    laplace_coefficients: &[f64],
    targets: &[[f64; 3]],
) -> (Vec<C>, Vec<[C; 3]>) {
    let level = Level::new(k, HALF);
    let phase = path.phase(k);
    let len = (P + 1) * (P + 1);
    let mut mapped = vec![C::new(0.0, 0.0); len];
    for n in 0..=P {
        for (m, i) in slots(n) {
            let d = if multipole {
                map_multipole(phase, n, m)
            } else {
                map_local(phase, n, m)
            };
            mapped[i] = d * laplace_coefficients[i];
        }
    }
    let mut ws = Workspace::new(P);
    let frame = frame();
    let mut potential = vec![C::new(0.0, 0.0); targets.len()];
    let mut gradient = vec![[C::new(0.0, 0.0); 3]; targets.len()];
    match path {
        Path::Complex(_) => {
            let mut stored: Vec<f64> = mapped.iter().map(|c| c.re).collect();
            stored.extend(mapped.iter().map(|c| c.im));
            if multipole {
                leaf::m2p(
                    P,
                    &level,
                    &frame,
                    &stored,
                    targets,
                    &mut ws,
                    &mut potential,
                    Some(&mut gradient),
                );
            } else {
                leaf::l2p(
                    P,
                    &level,
                    &frame,
                    &stored,
                    targets,
                    &mut ws,
                    &mut potential,
                    Some(&mut gradient),
                );
            }
        }
        Path::Yukawa => {
            let stored: Vec<f64> = mapped
                .iter()
                .map(|c| {
                    assert_eq!(c.im, 0.0);
                    c.re
                })
                .collect();
            let mut phi = vec![0.0; targets.len()];
            let mut grad = vec![[0.0; 3]; targets.len()];
            if multipole {
                leaf::m2p_yukawa(
                    P,
                    &level,
                    &frame,
                    &stored,
                    targets,
                    &mut ws,
                    &mut phi,
                    Some(&mut grad),
                );
            } else {
                leaf::l2p_yukawa(
                    P,
                    &level,
                    &frame,
                    &stored,
                    targets,
                    &mut ws,
                    &mut phi,
                    Some(&mut grad),
                );
            }
            potential = complexify(&phi);
            gradient = grad.iter().map(|g| g.map(|x| C::new(x, 0.0))).collect();
        }
    }
    (potential, gradient)
}

fn check_evaluation(multipole: bool) {
    let pts = points();
    let a = circumradius();
    let (sources, charges, targets) = if multipole {
        (&pts.inside, &pts.charges_inside, &pts.targets_outside)
    } else {
        (&pts.outside, &pts.charges_outside, &pts.targets_inside)
    };
    let coefficients = laplace_form(multipole, sources, charges);
    let mut ws = LaplaceWorkspace::new(P);
    let mut phi_l = vec![0.0; targets.len()];
    let mut grad_l = vec![[0.0; 3]; targets.len()];
    if multipole {
        laplace::m2p(
            P,
            &frame(),
            &coefficients,
            targets,
            &mut ws,
            &mut phi_l,
            Some(&mut grad_l),
        );
    } else {
        laplace::l2p(
            P,
            &frame(),
            &coefficients,
            targets,
            &mut ws,
            &mut phi_l,
            Some(&mut grad_l),
        );
    }
    let v_max = targets
        .iter()
        .map(|&x| distance(x, CENTRE))
        .fold(0.0, f64::max);
    let name = if multipole { "M2P" } else { "L2P" };
    let (mut potential, mut gradient, mut real, mut real_gradient) = (
        Worst::default(),
        Worst::default(),
        Worst::default(),
        Worst::default(),
    );
    for path in PATHS {
        for ka in MODULI {
            let k = path.k(ka);
            let kn = k.norm();
            let t = if multipole {
                kn * (a + v_max) + (kn * v_max).powi(2)
            } else {
                ka + ka * ka / 6.0
            };
            let t_real = if multipole {
                (kn * v_max).powi(2)
            } else {
                1.1 * ka * ka / 6.0
            };
            let (phi, grad) = evaluate(path, k, multipole, &coefficients, targets);
            for (i, &x) in targets.iter().enumerate() {
                let (mut mp, mut mg) = (0.0, 0.0);
                for (&y, &q) in sources.iter().zip(charges.iter()) {
                    let rho = distance(y, CENTRE);
                    let d = (distance(x, CENTRE) - rho).abs();
                    mp += q.abs() / d;
                    mg += 2.0 * q.abs() / (d * d);
                }
                let label = || format!("{path:?} |k|a = {ka:e} target {i}");
                let ep = (phi[i] - phi_l[i]).norm() / mp;
                let eg = (0..3)
                    .map(|c| (grad[i][c] - grad_l[i][c]).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / mg;
                potential.record(ep / (t + ROUND_OFF), label);
                gradient.record(eg / (2.0 * t + ROUND_OFF), label);
                if path.is_real() {
                    let rp = (phi[i].re - phi_l[i]).abs() / mp;
                    let rg = (0..3)
                        .map(|c| (grad[i][c].re - grad_l[i][c]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / mg;
                    real.record(rp / (t_real + ROUND_OFF), label);
                    real_gradient.record(rg / (2.0 * t_real + ROUND_OFF), label);
                    if !multipole {
                        // Real k: L2P of real coefficients is real.
                        assert!(phi[i].im.abs() <= ROUND_OFF * mp, "{}", label());
                    }
                }
            }
        }
    }
    potential.check(1.0, &format!("{name} potential, error / bound"));
    gradient.check(1.0, &format!("{name} gradient, error / bound"));
    real.check(1.0, &format!("{name} real k, real part, error / bound"));
    real_gradient.check(1.0, &format!("{name} real k, real gradient, error / bound"));
}

#[test]
fn m2p_evaluation() {
    check_evaluation(true);
}

#[test]
fn l2p_evaluation() {
    check_evaluation(false);
}

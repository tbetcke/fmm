//! Exactness of M2M and L2L: M2M after P2M equals P2M at the output frame, L2P after
//! L2L equals L2P of the input expansion, and two translations in a row equal the
//! single translation between the end frames (CONVENTIONS §3.11: both are identities
//! of polynomials, exact to degree p).
//!
//! Each test also prints, for transparency, the strict error: relative to the norm of
//! the reference per degree (coefficients) or to the term magnitudes of the reference
//! L2P alone (potentials). It grows with p because the translations sum terms far
//! larger than their result; it is reported, not asserted.
//!
//! Degrees: every p ≤ 30 for the three frames of `common::frames` at its first centre
//! (one per radius), and p in {0, 1, 2, 3, 7, 12, 20, 30} for the other three.
//!
//! Frame pairs: the eight octant children of each parent frame of `common::frames`
//! (r′ = 2r and c − c′ = r (±1, ±1, ±1) for M2M; the reverse for L2L), and random
//! nested pairs with radius ratios in [0.1, 0.9] whose inner sphere lies inside the
//! outer one.

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Frame, Workspace, direct, leaf};

use crate::common::{
    EXACT_TOL, Kind, P_MAX, SplitMix64, Strict, Worst, degree_error, degree_norms, frames,
    l2l_scale, l2l_shift, len, m2m_scale, m2m_shift, nested, norm, octants, p2m_magnitudes, place,
    random_coefficients,
};

/// Degrees sampled for the frames that are not swept over every p.
const SAMPLED: [usize; 8] = [0, 1, 2, 3, 7, 12, 20, P_MAX];

/// The degrees tested for the `index`-th frame of `common::frames`: every p ≤ 30
/// (CONVENTIONS §3.9) for the frames at the first centre, one per radius, and the
/// degrees in `SAMPLED` for the others, to keep the debug run short.
fn degrees(index: usize) -> Vec<usize> {
    if index.is_multiple_of(2) {
        (0..=P_MAX).collect()
    } else {
        SAMPLED.to_vec()
    }
}

/// Inner and outer frame pairs with their degrees: the octant children of every test
/// frame, and three random nested frames of each.
fn pairs(rng: &mut SplitMix64) -> Vec<(Frame<f64>, Frame<f64>, Vec<usize>)> {
    let mut out = Vec::new();
    for (index, outer) in frames().into_iter().enumerate() {
        for inner in octants(&outer) {
            out.push((inner, outer, degrees(index)));
        }
        for _ in 0..3 {
            out.push((nested(&outer, rng), outer, degrees(index)));
        }
    }
    out
}

/// P2M of `sources` in `from`, then M2M to `to`; and P2M of the same sources in `to`.
pub fn m2m_after_p2m(
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
    ws: &mut Workspace<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut inner = vec![0.0; len(p)];
    leaf::p2m(p, from, sources, charges, ws, &mut inner);
    let mut translated = vec![0.0; len(p)];
    direct::m2m(p, from, to, ws, &inner, &mut translated);
    let mut want = vec![0.0; len(p)];
    leaf::p2m(p, to, sources, charges, ws, &mut want);
    (translated, want)
}

/// The per-degree M2M term magnitudes of P2M in `from` followed by M2M to `to`.
pub fn m2m_p2m_scale(
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
) -> Vec<f64> {
    let u: Vec<[f64; 3]> = sources.iter().map(|&y| from.scaled(y)).collect();
    let (b, rho) = m2m_shift(from, to);
    m2m_scale(&p2m_magnitudes(p, &u, charges), rho, norm(b))
}

#[test]
fn m2m_after_p2m_equals_p2m_at_parent() {
    // Error measure: multipole coefficients per degree in the orthonormal weighting Nₘ,
    // relative to the M2M term magnitudes Σⱼ |qⱼ| (ρ|uⱼ| + |b|)ⁿ (`common`).
    let mut rng = SplitMix64::new(0x7501);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("M2M after P2M vs P2M at parent, per degree");
    let strict = Strict::new("M2M after P2M, relative to the reference per degree");
    for (child, parent, degrees) in pairs(&mut rng) {
        let sources: Vec<[f64; 3]> = (0..10).map(|_| place(&child, rng.in_cube())).collect();
        let charges: Vec<f64> = (0..10).map(|_| rng.charge()).collect();
        for &p in &degrees {
            let (got, want) = m2m_after_p2m(p, &child, &parent, &sources, &charges, &mut ws);
            let scale = m2m_p2m_scale(p, &child, &parent, &sources, &charges);
            let e = worst.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
            assert!(e <= EXACT_TOL, "{child:?} -> {parent:?}, p = {p}: {e:.3e}");
            let reference = degree_norms(Kind::Multipole, p, &want);
            strict.update(p, degree_error(Kind::Multipole, p, &got, &want, &reference));
        }
    }
}

/// M2M(b → c) after M2M(a → b), and M2M(a → c), of `input` in frame `a`.
pub fn m2m_chain(
    p: usize,
    [a, b, c]: [&Frame<f64>; 3],
    input: &[f64],
    ws: &mut Workspace<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut middle = vec![0.0; len(p)];
    direct::m2m(p, a, b, ws, input, &mut middle);
    let mut twice = vec![0.0; len(p)];
    direct::m2m(p, b, c, ws, &middle, &mut twice);
    let mut once = vec![0.0; len(p)];
    direct::m2m(p, a, c, ws, input, &mut once);
    (twice, once)
}

/// Term magnitudes of the chain a → b → c of M2M on `input` (they bound those of the
/// single M2M a → c; `common`).
pub fn m2m_chain_scale(p: usize, [a, b, c]: [&Frame<f64>; 3], input: &[f64]) -> Vec<f64> {
    let (b1, rho1) = m2m_shift(a, b);
    let (b2, rho2) = m2m_shift(b, c);
    let first = m2m_scale(&degree_norms(Kind::Multipole, p, input), rho1, norm(b1));
    m2m_scale(&first, rho2, norm(b2))
}

#[test]
fn m2m_composes() {
    // Error measure: multipole coefficients per degree in the orthonormal weighting Nₘ,
    // relative to the M2M term magnitudes of the two-step chain on the random input.
    let mut rng = SplitMix64::new(0x7502);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("M2M composition, per degree");
    let strict = Strict::new("M2M composition, relative to the reference per degree");
    for (index, c) in frames().into_iter().enumerate() {
        let mut chains = Vec::new();
        // Grandchild → child → parent in octants, and random nested chains.
        for b in octants(&c) {
            let a = octants(&b)[(rng.next_u64() % 8) as usize];
            chains.push([a, b, c]);
        }
        for _ in 0..4 {
            let b = nested(&c, &mut rng);
            chains.push([nested(&b, &mut rng), b, c]);
        }
        for [a, b, c] in chains {
            for p in degrees(index) {
                let input = random_coefficients(Kind::Multipole, p, &mut rng);
                let (twice, once) = m2m_chain(p, [&a, &b, &c], &input, &mut ws);
                let scale = m2m_chain_scale(p, [&a, &b, &c], &input);
                let e = worst.update(degree_error(Kind::Multipole, p, &twice, &once, &scale));
                assert!(e <= EXACT_TOL, "{a:?} -> {b:?} -> {c:?}, p = {p}: {e:.3e}");
                let reference = degree_norms(Kind::Multipole, p, &once);
                strict.update(
                    p,
                    degree_error(Kind::Multipole, p, &twice, &once, &reference),
                );
            }
        }
    }
}

/// L2P of `local` in `from` at `targets`, and L2P of its L2L to `to`.
pub fn l2p_after_l2l(
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    local: &[f64],
    targets: &[[f64; 3]],
    ws: &mut Workspace<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut translated = vec![0.0; len(p)];
    direct::l2l(p, from, to, ws, local, &mut translated);
    let mut got = vec![0.0; targets.len()];
    leaf::l2p(p, to, &translated, targets, ws, &mut got, None);
    let mut want = vec![0.0; targets.len()];
    leaf::l2p(p, from, local, targets, ws, &mut want, None);
    (got, want)
}

/// The sum of term magnitudes of L2L from `from` to `to` followed by L2P at `x`:
/// (1/r) Σₙ ‖L̃ₙ‖ (|t| + σ|v′|)ⁿ, with v′ = (x − c′)/r′ (`common`).
pub fn l2l_l2p_scale(from: &Frame<f64>, to: &Frame<f64>, magnitudes: &[f64], x: [f64; 3]) -> f64 {
    let (t, sigma) = l2l_shift(from, to);
    let reach = norm(t) + sigma * norm(to.scaled(x));
    let sum: f64 = magnitudes
        .iter()
        .enumerate()
        .map(|(n, a)| a * reach.powi(n as i32))
        .sum();
    sum / from.radius
}

/// The sum of term magnitudes of L2P of `local` in `frame` at `x`,
/// (1/r) Σᵢ wᵢ |L̃ᵢ Rᵢ(v)| over the slots of real storage, with wᵢ = 1 for m = 0 and 2
/// otherwise (the doubling rule of CONVENTIONS §3.6).
fn l2p_term_magnitudes(p: usize, frame: &Frame<f64>, local: &[f64], x: [f64; 3]) -> f64 {
    let layout = Layout::new(p);
    let mut values = vec![0.0; layout.len()];
    harmonics::regular(p, frame.scaled(x), &mut values);
    let sum: f64 = (0..layout.len())
        .map(|i| {
            let w = if layout.nm(i).1 == 0 { 1.0 } else { 2.0 };
            w * (local[i] * values[i]).abs()
        })
        .sum();
    sum / frame.radius
}

#[test]
fn l2l_is_exact_on_local_polynomials() {
    // Error measure: potential at targets inside the output box, relative to the sum
    // of term magnitudes of L2L followed by L2P, (1/r) Σₙ ‖L̃ₙ‖ (|t| + σ|v′|)ⁿ.
    let mut rng = SplitMix64::new(0x7503);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("L2P after L2L vs L2P, relative to term magnitudes");
    let strict = Strict::new("L2P after L2L, relative to the reference L2P term magnitudes");
    for (child, parent, degrees) in pairs(&mut rng) {
        let targets: Vec<[f64; 3]> = (0..10).map(|_| place(&child, rng.in_cube())).collect();
        for &p in &degrees {
            let local = random_coefficients(Kind::Local, p, &mut rng);
            let magnitudes = degree_norms(Kind::Local, p, &local);
            let (got, want) = l2p_after_l2l(p, &parent, &child, &local, &targets, &mut ws);
            for ((g, w), &x) in got.iter().zip(&want).zip(&targets) {
                let scale = l2l_l2p_scale(&parent, &child, &magnitudes, x);
                let e = worst.update((g - w).abs() / scale);
                assert!(e <= EXACT_TOL, "{parent:?} -> {child:?}, p = {p}: {e:.3e}");
                let reference = l2p_term_magnitudes(p, &parent, &local, x);
                strict.update(p, (g - w).abs() / reference);
            }
        }
    }
}

/// L2L(b → c) after L2L(a → b), and L2L(a → c), of `input` in frame `a`.
pub fn l2l_chain(
    p: usize,
    [a, b, c]: [&Frame<f64>; 3],
    input: &[f64],
    ws: &mut Workspace<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut middle = vec![0.0; len(p)];
    direct::l2l(p, a, b, ws, input, &mut middle);
    let mut twice = vec![0.0; len(p)];
    direct::l2l(p, b, c, ws, &middle, &mut twice);
    let mut once = vec![0.0; len(p)];
    direct::l2l(p, a, c, ws, input, &mut once);
    (twice, once)
}

/// Term magnitudes of the chain a → b → c of L2L on `input` (they bound those of the
/// single L2L a → c; `common`).
pub fn l2l_chain_scale(p: usize, [a, b, c]: [&Frame<f64>; 3], input: &[f64]) -> Vec<f64> {
    let (t1, sigma1) = l2l_shift(a, b);
    let (t2, sigma2) = l2l_shift(b, c);
    let first = l2l_scale(&degree_norms(Kind::Local, p, input), sigma1, norm(t1));
    l2l_scale(&first, sigma2, norm(t2))
}

#[test]
fn l2l_composes() {
    // Error measure: local coefficients per degree in the orthonormal weighting Nₘ/Sₘ,
    // relative to the L2L term magnitudes of the two-step chain on the random input.
    let mut rng = SplitMix64::new(0x7504);
    let mut ws = Workspace::new(P_MAX);
    let worst = Worst::new("L2L composition, per degree");
    let strict = Strict::new("L2L composition, relative to the reference per degree");
    for (index, a) in frames().into_iter().enumerate() {
        let mut chains = Vec::new();
        // Parent → child → grandchild in octants, and random nested chains.
        for b in octants(&a) {
            let c = octants(&b)[(rng.next_u64() % 8) as usize];
            chains.push([a, b, c]);
        }
        for _ in 0..4 {
            let b = nested(&a, &mut rng);
            chains.push([a, b, nested(&b, &mut rng)]);
        }
        for [a, b, c] in chains {
            for p in degrees(index) {
                let input = random_coefficients(Kind::Local, p, &mut rng);
                let (twice, once) = l2l_chain(p, [&a, &b, &c], &input, &mut ws);
                let scale = l2l_chain_scale(p, [&a, &b, &c], &input);
                let e = worst.update(degree_error(Kind::Local, p, &twice, &once, &scale));
                assert!(e <= EXACT_TOL, "{a:?} -> {b:?} -> {c:?}, p = {p}: {e:.3e}");
                let reference = degree_norms(Kind::Local, p, &once);
                strict.update(p, degree_error(Kind::Local, p, &twice, &once, &reference));
            }
        }
    }
}

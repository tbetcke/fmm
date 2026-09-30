//! Physical chains: P2M, then the M2L table, then L2P, against the same chain with
//! `direct::m2l`, and against the direct sum within the truncation bound of
//! CONVENTIONS §3.11 as the Phase 1 T5 test measures it.
//!
//! # Truncation bound
//!
//! Restated from fmm-ref/tests/direct/m2l.rs, whose module documentation derives it.
//! For one unit charge at y and a target at x, with u = (y − c)/r in the source frame
//! (c, r), v′ = (x − c′)/r′ in the target frame (c′, r′), b = (c′ − c)/r and σ = r′/r,
//! the error of P2M, M2L and L2P at degree p is at most E₁ + E₂:
//!
//! - input truncation, E₁ = (1/r′) T(ρ) Σⱼ≤ₚ √(2j + 1) σʲ⁺¹ |v′|ʲ / (|b| − ρ)ʲ⁺¹ with
//!   T(ρ) = Σₙ>ₚ (2n + 1) (|u|/ρ)ⁿ, for ρ = |u| |b| / (|u| + σ|v′|) (or (|u| + |b|)/2
//!   if v′ = 0; E₁ = 0 if u = 0);
//! - output truncation, E₂ = (a/d)ᵖ⁺¹ / (d − a) with a = |x − c′| and d = |y − c′|.
//!
//! The bound of several charges is Σₛ |qₛ| (E₁ + E₂). It needs |u| + σ|v′| < |b|,
//! which holds for every V-list pair: |u|, |v′| ≤ √3, σ = 1 and |b| ≥ 4. Rounding adds
//! a floor of `FLOOR` times the potential magnitude Σₛ |qₛ| / |x − yₛ|.

use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_ref::{Frame, Workspace, direct, leaf};
use nd_fmm_tables::MatrixSet;
use nd_fmm_tables::geometry::{m2l_frames, m2l_offset_index, m2l_offsets};

use crate::common::{
    CHAIN_TOL, DEBUG_DEGREES, DIRECT_SUM_OFFSETS, DYADIC, P_SUBSET, SplitMix64, Worst, apply,
    degree, l2p_terms, len, m2l, place, random_pair, subset, terms,
};

/// Rounding floor of the chain against the direct sum, relative to the potential
/// magnitude (as in Phase 1).
const FLOOR: f64 = 1e-14;

/// Levels of the chain tests, with the canonical frames as `None`.
const CHAIN_LEVELS: [Option<u32>; 4] = [None, Some(2), Some(9), Some(16)];

/// The (source, target) frames of offset `d` (table index `index`): the canonical
/// frames of §3.12, or a random V-list pair on `level` of the dyadic domain.
fn frames(
    level: Option<u32>,
    index: usize,
    d: [i64; 3],
    rng: &mut SplitMix64,
) -> (Frame<f64>, Frame<f64>) {
    match level {
        None => m2l_frames(index),
        Some(l) => {
            let (source, target) = random_pair(l, d, rng);
            (DYADIC.frame(l, source), DYADIC.frame(l, target))
        }
    }
}

/// P2M in `source` of the charges, the M2L `m2l` from the multipole to a zeroed local,
/// then L2P in `target` at `targets`, all at degree p. Returns the multipole and the
/// potentials.
fn chain(
    p: usize,
    (source, target): (&Frame<f64>, &Frame<f64>),
    (sources, charges): (&[[f64; 3]], &[f64]),
    targets: &[[f64; 3]],
    ws: &mut Workspace<f64>,
    m2l: impl FnOnce(&[f64], &mut Workspace<f64>) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut multipole = vec![0.0; len(p)];
    leaf::p2m(p, source, sources, charges, ws, &mut multipole);
    let local = m2l(&multipole, ws);
    let mut potential = vec![0.0; targets.len()];
    leaf::l2p(p, target, &local, targets, ws, &mut potential, None);
    (multipole, potential)
}

#[test]
fn p2m_table_l2p_equals_the_chain_with_direct_m2l() {
    // Error measure: potential at each target in the target box, relative to the L2P
    // terms (1/r) Σᵢ wᵢ |Rᵢ(v)| τᵢ of the table's terms τ = |A| |M̃| (`support`); both
    // chains share P2M and differ only in the M2L summation order. All 316 offsets at
    // p ≤ 8, at the canonical frames and on levels 2, 9 and 16 of the dyadic domain.
    let mut rng = SplitMix64::new(0x4d21_0101);
    let mut ws = Workspace::new(DEBUG_DEGREES[DEBUG_DEGREES.len() - 1]);
    let worst = Worst::new("P2M, M2L table, L2P vs chain with direct M2L, relative to terms");
    for level in CHAIN_LEVELS {
        for (index, d) in m2l_offsets().into_iter().enumerate() {
            let (source, target) = frames(level, index, d, &mut rng);
            let sources: Vec<[f64; 3]> = (0..6).map(|_| place(&source, rng.in_cube())).collect();
            let charges: Vec<f64> = (0..6).map(|_| rng.charge()).collect();
            let targets: Vec<[f64; 3]> = (0..6).map(|_| place(&target, rng.in_cube())).collect();
            for p in DEBUG_DEGREES {
                let set = m2l(p).matrices();
                let (multipole, got) = chain(
                    p,
                    (&source, &target),
                    (&sources, &charges),
                    &targets,
                    &mut ws,
                    |m, _| apply(set, index, m),
                );
                let (_, want) = chain(
                    p,
                    (&source, &target),
                    (&sources, &charges),
                    &targets,
                    &mut ws,
                    |m, ws| {
                        let mut local = vec![0.0; len(p)];
                        direct::m2l(p, &source, &target, ws, m, &mut local);
                        local
                    },
                );
                let tau = terms(set, index, &multipole);
                for ((g, w), &x) in got.iter().zip(&want).zip(&targets) {
                    let e = worst.update((g - w).abs() / l2p_terms(p, &target, &tau, x));
                    assert!(e <= CHAIN_TOL, "{level:?}, d = {d:?}, p = {p}: {e:.3e}");
                }
            }
        }
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    core::array::from_fn(|i| a[i] - b[i])
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The truncation bound E₁ + E₂ of the module documentation for a unit charge at `y`
/// and a target at `x`.
fn bound(p: usize, source: &Frame<f64>, target: &Frame<f64>, y: [f64; 3], x: [f64; 3]) -> f64 {
    let (b, sigma) = (source.scaled(target.centre), target.radius / source.radius);
    let (u, v, b) = (norm(source.scaled(y)), norm(target.scaled(x)), norm(b));
    assert!(
        u + sigma * v < b,
        "the bound needs |u| + σ|v′| < |b|: {u} + {sigma} · {v} vs {b}"
    );
    let input = if u == 0.0 {
        0.0
    } else {
        let rho = if v > 0.0 {
            u * b / (u + sigma * v)
        } else {
            (u + b) / 2.0
        };
        let ratio = u / rho;
        let big_n = (p + 1) as f64;
        let tail = ratio.powi(p as i32 + 1)
            * ((2.0 * big_n + 1.0) / (1.0 - ratio) + 2.0 * ratio / (1.0 - ratio).powi(2));
        let sum: f64 = (0..=p)
            .map(|j| {
                ((2 * j + 1) as f64).sqrt() * sigma.powi(j as i32 + 1) * v.powi(j as i32)
                    / (b - rho).powi(j as i32 + 1)
            })
            .sum();
        tail * sum / target.radius
    };
    let a = norm(sub(x, target.centre));
    let d = norm(sub(y, target.centre));
    let output = (a / d).powi(p as i32 + 1) / (d - a);
    input + output
}

#[test]
fn p2m_table_l2p_matches_the_direct_sum_within_the_truncation_bound() {
    // Error measure: potential at each target against `direct_sum`, absolute, within
    // the truncation bound of the module documentation plus a floor of 1e-14 times
    // Σₛ |qₛ| / |x − yₛ|, as in fmm-ref/tests/direct/m2l.rs. Offsets (2, 0, 0),
    // (2, 2, 2) and (3, 3, 3) at p ∈ {4, 8, 12}, at the canonical frames and on level 2
    // of the dyadic domain; 8 sources and 8 targets uniform in their boxes, plus one
    // pair at the facing corners shrunk to 95 %, where the bound is least slack.
    let mut rng = SplitMix64::new(0x4d21_0102);
    let mut ws = Workspace::new(P_SUBSET);
    let attained = Worst::new("M2L table chain vs direct sum, error / (bound + floor)");
    let subset = subset();
    for p in [4, 8, P_SUBSET] {
        let relative = Worst::new(match p {
            4 => "M2L table chain vs direct sum, p = 4, relative to Σ|q|/|x − y|",
            8 => "M2L table chain vs direct sum, p = 8, relative to Σ|q|/|x − y|",
            _ => "M2L table chain vs direct sum, p = 12, relative to Σ|q|/|x − y|",
        });
        let relative_bound = Worst::new(match p {
            4 => "M2L table chain, p = 4, bound relative to Σ|q|/|x − y|",
            8 => "M2L table chain, p = 8, bound relative to Σ|q|/|x − y|",
            _ => "M2L table chain, p = 12, bound relative to Σ|q|/|x − y|",
        });
        for d in DIRECT_SUM_OFFSETS {
            let index = m2l_offset_index(d).unwrap();
            let (set, position): (&MatrixSet<f64>, usize) = if p == P_SUBSET {
                (&subset.matrices, subset.position(index))
            } else {
                (m2l(p).matrices(), index)
            };
            assert_eq!(degree(set), p);
            for level in [None, Some(2)] {
                let (source, target) = frames(level, index, d, &mut rng);
                // The facing corners: towards the target along each axis that d moves,
                // and +1 along the others.
                let toward = d.map(|t| if t < 0 { -0.95 } else { 0.95 });
                let away = d.map(|t| if t > 0 { -0.95 } else { 0.95 });
                let mut sources: Vec<[f64; 3]> =
                    (0..8).map(|_| place(&source, rng.in_cube())).collect();
                let mut targets: Vec<[f64; 3]> =
                    (0..8).map(|_| place(&target, rng.in_cube())).collect();
                sources.push(place(&source, toward));
                targets.push(place(&target, away));
                let charges: Vec<f64> = (0..sources.len()).map(|_| rng.charge()).collect();

                let (_, got) = chain(
                    p,
                    (&source, &target),
                    (&sources, &charges),
                    &targets,
                    &mut ws,
                    |m, _| apply(set, position, m),
                );
                let mut exact = vec![0.0; targets.len()];
                direct_sum(&sources, &charges, &targets, &mut exact, None);
                for ((&g, &e), &x) in got.iter().zip(&exact).zip(&targets) {
                    let mut magnitude = 0.0;
                    let mut limit = 0.0;
                    for (&y, &q) in sources.iter().zip(&charges) {
                        magnitude += q.abs() / norm(sub(x, y));
                        limit += q.abs() * bound(p, &source, &target, y, x);
                    }
                    let error = (g - e).abs();
                    let allowed = limit + FLOOR * magnitude;
                    assert!(
                        error <= allowed,
                        "{level:?}, d = {d:?}, p = {p}: error {error:.3e} exceeds bound \
                         {limit:.3e} + floor {:.3e}",
                        FLOOR * magnitude
                    );
                    attained.update(error / allowed);
                    relative.update(error / magnitude);
                    relative_bound.update(limit / magnitude);
                }
            }
        }
    }
    assert!(attained.get() > 0.0);
}

//! Chains against the direct sum: P2M at a source leaf, M2L to a target leaf, L2P at
//! the target points, all by the per-pair kernels of `LaplaceOperator`, against
//! `p2p::direct_sum` over the absolute coordinates, within the truncation bound of
//! CONVENTIONS §3.11 as the Phase 1 T5 test measures it.
//!
//! The chain's φ̂ is converted to 1/|x − y| units, φ = φ̂ / r_t; 4π is not involved.
//!
//! # Truncation bound
//!
//! Restated from fmm-tables/tests/m2l/chains.rs (derived in fmm-ref/tests/direct/m2l.rs).
//! For one unit charge at y and a target at x, with u = (y − c)/r in the source frame
//! (c, r), v′ = (x − c′)/r′ in the target frame (c′, r′), b = (c′ − c)/r and σ = r′/r,
//! the error of P2M, M2L and L2P at degree p is at most E₁ + E₂:
//!
//! - input truncation, E₁ = (1/r′) T(ρ) Σⱼ≤ₚ √(2j + 1) σʲ⁺¹ |v′|ʲ / (|b| − ρ)ʲ⁺¹ with
//!   T(ρ) = Σₙ>ₚ (2n + 1) (|u|/ρ)ⁿ, for ρ = |u| |b| / (|u| + σ|v′|) (or (|u| + |b|)/2
//!   if v′ = 0; E₁ = 0 if u = 0);
//! - output truncation, E₂ = (a/d)ᵖ⁺¹ / (d − a) with a = |x − c′| and d = |y − c′|.
//!
//! The bound of several charges is Σₛ |qₛ| (E₁ + E₂). Rounding adds a floor of 1e-14
//! times the potential magnitude Σₛ |qₛ| / |x − yₛ|, as in Phase 1.
//!
//! The frames are the absolute ones of the dyadic domain, exact on every level, and the
//! points lie on the grid of `common` (the facing corners at ±996147 / 2²⁰ ≈ ±0.95), so
//! the leaf-scaled coordinates are exact.

use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_ref::Frame;
use nd_fmm_ref::p2p::direct_sum;
use nd_octree::MortonKey;

use crate::common::{
    LEVELS, SplitMix64, Worst, absolute_frame, dyadic_domain, len, norm, operator, random_pair,
    source_chunk, sub, target_chunk,
};

/// The offsets of the chains.
const OFFSETS: [[i64; 3]; 3] = [[2, 0, 0], [2, 2, 2], [3, 3, 3]];

/// The rounding floor relative to the potential magnitude.
const FLOOR: f64 = 1e-14;

/// The grid value nearest 0.95: 996147 / 2²⁰.
const CORNER: f64 = 996147.0 / 1048576.0;

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

/// Eight random grid points of `key` and one at the corner `corner` (in units of the
/// box): absolute and leaf-scaled coordinates.
fn chain_points(
    key: MortonKey,
    corner: [f64; 3],
    rng: &mut SplitMix64,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let frame = absolute_frame(key, &dyadic_domain());
    let mut u: Vec<[f64; 3]> = (0..8).map(|_| [0; 3].map(|_| rng.grid())).collect();
    u.push(corner);
    let x = u
        .iter()
        .map(|u| core::array::from_fn(|k| frame.centre[k] + frame.radius * u[k]))
        .collect();
    (x, u)
}

#[test]
fn p2m_m2l_l2p_matches_the_direct_sum_within_the_truncation_bound() {
    // Error measure: potential at each target against `direct_sum`, absolute, within
    // the truncation bound plus a floor of 1e-14 times Σₛ |qₛ| / |x − yₛ| (module
    // documentation). Offsets (2, 0, 0), (2, 2, 2) and (3, 3, 3) on levels 2, 9 and 16
    // of the dyadic domain, at p = 4 and 8 with every strategy and at p = 12 with
    // `Classes` and `Rotation` (the dense M2L tables at p = 12 take too long to build
    // in the debug run; `Auto` resolves to `Rotation` there). Eight sources and eight
    // targets on the grid of their boxes, plus one pair at the facing corners, where
    // the bound is least slack.
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7821);
    let attained = Worst::new("chain vs direct sum, error / (bound + floor)");
    for (p, strategies) in [
        (
            4,
            &[
                M2lStrategy::Dense,
                M2lStrategy::Classes,
                M2lStrategy::Rotation,
            ][..],
        ),
        (
            8,
            &[
                M2lStrategy::Dense,
                M2lStrategy::Classes,
                M2lStrategy::Rotation,
            ],
        ),
        (12, &[M2lStrategy::Classes, M2lStrategy::Rotation]),
    ] {
        let relative = Worst::new(format!(
            "chain vs direct sum, p = {p}, relative to Σ|q|/|x − y|"
        ));
        let relative_bound = Worst::new(format!("chain, p = {p}, bound relative to Σ|q|/|x − y|"));
        for &strategy in strategies {
            let mut op = operator(strategy, p, 0);
            for d in OFFSETS {
                for level in LEVELS {
                    let (source, target) = random_pair(level, d, &mut rng);
                    let (from, to) = (
                        absolute_frame(source, &domain),
                        absolute_frame(target, &domain),
                    );
                    // The facing corners: towards the target along each axis that d
                    // moves, and +1 along the others.
                    let toward = d.map(|t| if t < 0 { -CORNER } else { CORNER });
                    let away = d.map(|t| if t > 0 { -CORNER } else { CORNER });
                    let (y, u_s) = chain_points(source, toward, &mut rng);
                    let (x, u_t) = chain_points(target, away, &mut rng);
                    let q = rng.charges(y.len());

                    let mut multipole = vec![0.0; len(p)];
                    op.p2m_leaf(&source_chunk(&u_s, &q), &mut multipole);
                    let mut local = vec![0.0; len(p)];
                    op.m2l_pair(source, target, &multipole, &mut local);
                    let mut output = vec![0.0; 4 * x.len()];
                    op.l2p_leaf(&local, &target_chunk(&u_t), &mut output);

                    let mut exact = vec![0.0; x.len()];
                    direct_sum(&y, &q, &x, &mut exact, None);
                    for (j, &xj) in x.iter().enumerate() {
                        let got = output[j] / to.radius;
                        let mut magnitude = 0.0;
                        let mut limit = 0.0;
                        for (&yk, &qk) in y.iter().zip(&q) {
                            magnitude += qk.abs() / norm(sub(xj, yk));
                            limit += qk.abs() * bound(p, &from, &to, yk, xj);
                        }
                        let error = (got - exact[j]).abs();
                        let allowed = limit + FLOOR * magnitude;
                        assert!(
                            error <= allowed,
                            "{strategy:?}, level {level}, d = {d:?}, p = {p}: error {error:.3e} \
                             exceeds bound {limit:.3e} + floor {:.3e}",
                            FLOOR * magnitude
                        );
                        attained.update(error / allowed);
                        relative.update(error / magnitude);
                        relative_bound.update(limit / magnitude);
                    }
                }
            }
        }
    }
    assert!(attained.get() > 0.0);
}

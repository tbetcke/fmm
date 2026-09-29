//! M2L against the direct sum, within a truncation bound, and the structure of M2L.
//!
//! # Truncation bound
//!
//! P2M, then M2L, then L2P, all at degree p, approximates the potential
//! Σₛ qₛ / |x − yₛ| at a target x. Take one unit charge at y, with u = (y − c)/r in the
//! source frame (c, r), v′ = (x − c′)/r′ in the target frame (c′, r′), the M2L shift
//! b = (c′ − c)/r and σ = r′/r. Let L̃′ be the M2L output from the input truncated at p,
//! and L̃* the P2L coefficients of the charge in the target frame, which the M2L sum
//! gives with the input not truncated (CONVENTIONS §3.11, "M2L", Truncation). Then
//!
//! φ − φₚ = (1/r′) Σⱼ≤ₚ Σᵢ (L̃*ⱼⁱ − L̃′ⱼⁱ) Rⱼⁱ(v′) + (1/r′) Σⱼ>ₚ Σᵢ L̃*ⱼⁱ Rⱼⁱ(v′).
//!
//! 1. *Input truncation.* §3.11 bounds each |L̃*ⱼⁱ − L̃′ⱼⁱ|, for any ρ in (|u|, |b|), by
//!    σʲ⁺¹ √((j − |i|)! (j + |i|)!) / (|b| − ρ)ʲ⁺¹ · T(ρ), with
//!    T(ρ) = Σₙ>ₚ (2n + 1) (|u|/ρ)ⁿ = xᴺ ((2N + 1)/(1 − x) + 2x/(1 − x)²) for
//!    x = |u|/ρ and N = p + 1. By Cauchy–Schwarz over the 2j + 1 orders and
//!    Σᵢ (j − |i|)! (j + |i|)! |Rⱼⁱ(v′)|² = |v′|²ʲ (see `common`),
//!    Σᵢ √((j − |i|)! (j + |i|)!) |Rⱼⁱ(v′)| ≤ √(2j + 1) |v′|ʲ. So the first sum is at
//!    most
//!
//!    E₁ = (1/r′) T(ρ) Σⱼ≤ₚ √(2j + 1) σʲ⁺¹ |v′|ʲ / (|b| − ρ)ʲ⁺¹.
//!
//!    We take ρ = |u| |b| / (|u| + σ|v′|), which makes the two ratios |u|/ρ and
//!    σ|v′|/(|b| − ρ) equal to (|u| + σ|v′|)/|b|; it lies in (|u|, |b|) because the
//!    tested geometry has |u| + σ|v′| < |b|. For v′ = 0 (a target at the output
//!    centre) that choice gives ρ = |b|, so we take ρ = (|u| + |b|)/2 instead; then
//!    only j = 0 contributes to the sum. For u = 0 the input is a monopole and E₁ = 0.
//! 2. *Output truncation.* The second sum is the tail of the local expansion of
//!    1/|x − y| about c′, the Legendre series Σⱼ aʲ/dʲ⁺¹ Pⱼ(cos γ) with
//!    a = |x − c′| < d = |y − c′|. With |Pⱼ| ≤ 1 it is at most
//!
//!    E₂ = (a/d)ᵖ⁺¹ / (d − a).
//!
//! The bound is Σₛ |qₛ| (E₁ + E₂). Both terms decay like ((|u| + σ|v′|)/|b|)ᵖ, up to
//! polynomial factors; for the V list (σ = 1, |b| ≥ 4, |u|, |v′| ≤ √3) that is at worst
//! (2√3/4)ᵖ ≈ 0.87ᵖ, attained only by opposite box corners.
//!
//! Rounding adds a floor of `FLOOR` times the potential magnitude Σₛ |qₛ| / |x − yₛ|,
//! which matters only where the bound falls below rounding.

use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_ref::{Frame, Workspace, direct, leaf};

use crate::common::{
    EXACT_TOL, Kind, P_MAX_M2L, SQRT3, SplitMix64, Worst, degree_error, degree_norms, l2l_shift,
    len, m2l_scale, norm, p2m_magnitudes, place, random_coefficients, sub,
};

/// Rounding floor of the M2L chain, relative to the potential magnitude.
const FLOOR: f64 = 1e-14;

/// The 316 V-list offsets d in {−3..3}³ \ {−1..1}³ (design §2.5).
pub fn offsets() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for x in -3i32..=3 {
        for y in -3i32..=3 {
            for z in -3i32..=3 {
                if x.abs().max(y.abs()).max(z.abs()) > 1 {
                    out.push([x, y, z].map(f64::from));
                }
            }
        }
    }
    out
}

/// The truncation bound E₁ + E₂ of the module documentation for a unit charge at `y`
/// and a target at `x`.
pub fn bound(p: usize, source: &Frame<f64>, target: &Frame<f64>, y: [f64; 3], x: [f64; 3]) -> f64 {
    let (b, sigma) = l2l_shift(source, target);
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

/// P2M in `source`, M2L to `target` and L2P at `targets`, all at degree p.
pub fn p2m_m2l_l2p(
    p: usize,
    source: &Frame<f64>,
    target: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    ws: &mut Workspace<f64>,
) -> Vec<f64> {
    let mut multipole = vec![0.0; len(p)];
    leaf::p2m(p, source, sources, charges, ws, &mut multipole);
    let mut local = vec![0.0; len(p)];
    direct::m2l(p, source, target, ws, &multipole, &mut local);
    let mut potential = vec![0.0; targets.len()];
    leaf::l2p(p, target, &local, targets, ws, &mut potential, None);
    potential
}

/// One M2L configuration checked against the direct sum: its worst error and worst
/// bound, both relative to the potential magnitude Σₛ |qₛ| / |x − yₛ|, and the worst
/// ratio of error to bound plus floor.
pub struct Check {
    pub relative: f64,
    pub bound: f64,
    pub attained: f64,
}

/// Checks P2M-M2L-L2P against `direct_sum` at every target. Panics if an error exceeds
/// its bound plus floor.
pub fn check_against_direct_sum(
    p: usize,
    source: &Frame<f64>,
    target: &Frame<f64>,
    sources: &[[f64; 3]],
    charges: &[f64],
    targets: &[[f64; 3]],
    ws: &mut Workspace<f64>,
) -> Check {
    let got = p2m_m2l_l2p(p, source, target, sources, charges, targets, ws);
    let mut exact = vec![0.0; targets.len()];
    direct_sum(sources, charges, targets, &mut exact, None);
    let mut check = Check {
        relative: 0.0,
        bound: 0.0,
        attained: 0.0,
    };
    for ((&g, &e), &x) in got.iter().zip(&exact).zip(targets) {
        let mut magnitude = 0.0;
        let mut limit = 0.0;
        for (&y, &q) in sources.iter().zip(charges) {
            magnitude += q.abs() / norm(sub(x, y));
            limit += q.abs() * bound(p, source, target, y, x);
        }
        let error = (g - e).abs();
        let allowed = limit + FLOOR * magnitude;
        assert!(
            error <= allowed,
            "{source:?} -> {target:?}, p = {p}: error {error:.3e} exceeds bound {limit:.3e} \
             + floor {:.3e}",
            FLOOR * magnitude
        );
        check.relative = check.relative.max(error / magnitude);
        check.bound = check.bound.max(limit / magnitude);
        check.attained = check.attained.max(error / allowed);
    }
    check
}

/// Least-squares slope of log(y) against x.
fn log_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), &(x, y)| (sx + x, sy + y.ln()));
    let (mx, my) = (sx / n, sy / n);
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for &(x, y) in points {
        sxy += (x - mx) * (y.ln() - my);
        sxx += (x - mx) * (x - mx);
    }
    sxy / sxx
}

/// Exponential rate per degree of `values[p]`, p = 1..=`P_MAX_M2L`, fitted by least
/// squares on the logarithm.
fn rate(values: &[f64]) -> f64 {
    let points: Vec<(f64, f64)> = (1..=P_MAX_M2L).map(|p| (p as f64, values[p])).collect();
    log_slope(&points).exp()
}

#[test]
fn m2l_matches_direct_sum_on_the_v_list() {
    // Error measure: potential at each target against `direct_sum`, absolute, within
    // the truncation bound of the module documentation plus a floor of 1e-14 times
    // Σₛ |qₛ| / |x − yₛ|. Frames: same level (r′ = r), shift 2r · d for all 316
    // offsets; 8 sources and 8 targets uniform in their boxes; p = 0..=20.
    //
    // Decay: per offset, and over all offsets together, the worst error and the worst
    // bound over the targets, both relative to Σₛ |qₛ| / |x − yₛ|, as functions of p.
    // Their exponential rates in p, fitted by least squares over p = 1..=20, must
    // satisfy rate(error) ≤ rate(bound): the observed error decays at least at the
    // bound's rate.
    let mut rng = SplitMix64::new(0x7505);
    let mut ws = Workspace::new(P_MAX_M2L);
    let source = Frame::new([0.3, -1.1, 0.7], 0.37);
    let mut all_error = vec![0.0f64; P_MAX_M2L + 1];
    let mut all_bound = vec![0.0f64; P_MAX_M2L + 1];
    let attained = Worst::new("M2L V list, error / (bound + floor)");
    let rate_margin = Worst::new("M2L V list, per-offset rate(error) / rate(bound)");
    for d in offsets() {
        let target = Frame::new(place(&source, d.map(|c| 2.0 * c)), source.radius);
        let sources: Vec<[f64; 3]> = (0..8).map(|_| place(&source, rng.in_cube())).collect();
        let charges: Vec<f64> = (0..8).map(|_| rng.charge()).collect();
        let targets: Vec<[f64; 3]> = (0..8).map(|_| place(&target, rng.in_cube())).collect();
        let mut error = vec![0.0; P_MAX_M2L + 1];
        let mut limit = vec![0.0; P_MAX_M2L + 1];
        for p in 0..=P_MAX_M2L {
            let check = check_against_direct_sum(
                p, &source, &target, &sources, &charges, &targets, &mut ws,
            );
            attained.update(check.attained);
            (error[p], limit[p]) = (check.relative, check.bound);
            all_error[p] = all_error[p].max(check.relative);
            all_bound[p] = all_bound[p].max(check.bound);
        }
        let (rate_error, rate_bound) = (rate(&error), rate(&limit));
        assert!(
            rate_margin.update(rate_error / rate_bound) <= 1.0,
            "offset {d:?}: error decays at {rate_error:.4} per degree, slower than the \
             bound's {rate_bound:.4}"
        );
    }
    for p in 0..=P_MAX_M2L {
        eprintln!(
            "M2L V list p = {p:2}: worst relative error {:.3e}, worst relative bound {:.3e}",
            all_error[p], all_bound[p]
        );
    }
    let (rate_error, rate_bound) = (rate(&all_error), rate(&all_bound));
    eprintln!("M2L V list decay rate per degree: error {rate_error:.4}, bound {rate_bound:.4}");
    assert!(
        rate_error <= rate_bound,
        "error decays at {rate_error:.4} per degree, slower than the bound's {rate_bound:.4}"
    );
}

#[test]
fn m2l_matches_direct_sum_near_the_convergence_limit() {
    // Error measure: as `m2l_matches_direct_sum_on_the_v_list`. The frames are just
    // outside the convergence limit |b| > √3 (1 + σ) of CONVENTIONS §3.11, with σ in
    // {1, 1/2, 2}, and the sources and targets fill most of their boxes, so the
    // truncation error decays slowly. M2L does not check the separation; the chain
    // must still stay within its bound for p ≤ 20.
    let mut rng = SplitMix64::new(0x7506);
    let mut ws = Workspace::new(P_MAX_M2L);
    let attained = Worst::new("M2L near the limit, error / (bound + floor)");
    let relative = Worst::new("M2L near the limit at p = 20, relative error");
    let source = Frame::new([-0.4, 0.9, 1.3], 0.8);
    // (σ, direction of b, |b| / (√3 (1 + σ))).
    let cases: [(f64, [f64; 3], f64); 3] = [
        (1.0, [1.0, 1.0, 1.0], 3.6 / (2.0 * SQRT3)),
        (0.5, [1.0, 0.0, 0.0], 2.8 / (1.5 * SQRT3)),
        (2.0, [0.0, -1.0, 1.0], 5.5 / (3.0 * SQRT3)),
    ];
    for (sigma, direction, margin) in cases {
        let length = margin * SQRT3 * (1.0 + sigma);
        let b = direction.map(|c| c * length / norm(direction));
        let target = Frame::new(place(&source, b), sigma * source.radius);
        // Sources and targets in the boxes shrunk to 90 %, which keeps |u| + σ|v′| < |b|
        // as the bound requires; one pair sits at the facing corners.
        let unit = b.map(|c| c / length);
        let mut sources: Vec<[f64; 3]> = (0..7)
            .map(|_| place(&source, rng.in_cube().map(|c| 0.9 * c)))
            .collect();
        let mut targets: Vec<[f64; 3]> = (0..7)
            .map(|_| place(&target, rng.in_cube().map(|c| 0.9 * c)))
            .collect();
        sources.push(place(
            &source,
            unit.map(|c| 0.9 * c.signum() * (c != 0.0) as u8 as f64),
        ));
        targets.push(place(
            &target,
            unit.map(|c| -0.9 * c.signum() * (c != 0.0) as u8 as f64),
        ));
        let charges: Vec<f64> = (0..8).map(|_| rng.charge()).collect();
        for p in 0..=P_MAX_M2L {
            let check = check_against_direct_sum(
                p, &source, &target, &sources, &charges, &targets, &mut ws,
            );
            attained.update(check.attained);
            if p == P_MAX_M2L {
                relative.update(check.relative);
            }
        }
    }
}

/// Frame pairs for the M2L structure tests: V-list offsets on one level (σ = 1), and
/// shifts with σ in {1/2, 0.7, 1.7, 2} beyond the convergence limit √3 (1 + σ).
fn structure_pairs() -> Vec<(Frame<f64>, Frame<f64>)> {
    let source = Frame::new([0.3, -1.1, 0.7], 0.37);
    let mut out: Vec<(Frame<f64>, Frame<f64>)> = [
        [2.0, 0.0, 0.0],
        [0.0, 0.0, -2.0],
        [-3.0, 3.0, 3.0],
        [1.0, -2.0, 3.0],
        [0.0, 3.0, -1.0],
    ]
    .iter()
    .map(|d| {
        let target = Frame::new(place(&source, d.map(|c| 2.0 * c)), source.radius);
        (source, target)
    })
    .collect();
    for (sigma, b) in [
        (0.5, [3.0, -1.0, 1.5]),
        (0.7, [-1.0, 3.0, -1.5]),
        (1.7, [-2.0, 4.5, 3.0]),
        (2.0, [4.0, 4.0, -3.0]),
    ] {
        let target = Frame::new(place(&source, b), sigma * source.radius);
        out.push((source, target));
    }
    out
}

#[test]
fn m2l_of_a_unit_monopole_equals_p2l_of_a_unit_charge() {
    // Error measure: local coefficients per degree in the orthonormal weighting
    // Nₘ/Sₘ, relative to the weighted norm of the P2L result, (σ/|b|)ʲ⁺¹ (`common`).
    // With only slot 0 set, M2L has a single term per output and no truncation.
    let mut ws = Workspace::new(P_MAX_M2L);
    let worst = Worst::new("M2L of a unit monopole vs P2L, per degree");
    for (source, target) in structure_pairs() {
        let (b, sigma) = l2l_shift(&source, &target);
        for p in 0..=P_MAX_M2L {
            let mut monopole = vec![0.0; len(p)];
            monopole[0] = 1.0;
            let mut got = vec![0.0; len(p)];
            direct::m2l(p, &source, &target, &mut ws, &monopole, &mut got);
            let mut want = vec![0.0; len(p)];
            leaf::p2l(p, &target, &[source.centre], &[1.0], &mut ws, &mut want);
            let scale = m2l_scale(&degree_norms(Kind::Multipole, p, &monopole), sigma, norm(b));
            let e = worst.update(degree_error(Kind::Local, p, &got, &want, &scale));
            assert!(e <= EXACT_TOL, "{source:?} -> {target:?}, p = {p}: {e:.3e}");
        }
    }
}

#[test]
fn m2l_is_the_leading_block_of_a_higher_degree_m2l() {
    // Error measure: local coefficients per degree in the orthonormal weighting
    // Nₘ/Sₘ, relative to the M2L term magnitudes of the input (`common`). The input is
    // random, zero-padded from degree p to p + 5.
    let mut rng = SplitMix64::new(0x7507);
    let mut ws = Workspace::new(P_MAX_M2L + 5);
    let worst = Worst::new("M2L at p vs leading block of M2L at p + 5, per degree");
    for (source, target) in structure_pairs() {
        let (b, sigma) = l2l_shift(&source, &target);
        for p in 0..=P_MAX_M2L {
            let input = random_coefficients(Kind::Multipole, p, &mut rng);
            let mut got = vec![0.0; len(p)];
            direct::m2l(p, &source, &target, &mut ws, &input, &mut got);
            let mut padded = vec![0.0; len(p + 5)];
            padded[..len(p)].copy_from_slice(&input);
            let mut larger = vec![0.0; len(p + 5)];
            direct::m2l(p + 5, &source, &target, &mut ws, &padded, &mut larger);
            let scale = m2l_scale(&degree_norms(Kind::Multipole, p, &input), sigma, norm(b));
            let e = worst.update(degree_error(
                Kind::Local,
                p,
                &got,
                &larger[..len(p)],
                &scale,
            ));
            assert!(e <= EXACT_TOL, "{source:?} -> {target:?}, p = {p}: {e:.3e}");
        }
    }
}

#[test]
fn bound_is_finite_at_the_output_centre() {
    // A target exactly at c′ (v′ = 0) must get a finite bound, so that it can fail.
    let source = Frame::new([0.3, -1.1, 0.7], 0.37);
    let target = Frame::new(place(&source, [4.0, 0.0, 2.0]), source.radius);
    let y = place(&source, [0.5, -0.25, 0.75]);
    for p in [0, 1, 7, P_MAX_M2L] {
        let limit = bound(p, &source, &target, y, target.centre);
        assert!(limit.is_finite() && limit > 0.0, "p = {p}: {limit}");
    }
}

/// T(ρ) = Σₙ>ₚ (2n + 1) xⁿ = xᴺ ((2N + 1)/(1 − x) + 2x/(1 − x)²) with x = |u|/ρ and
/// N = p + 1 (CONVENTIONS §3.11, M2L truncation).
fn tail(p: usize, u: f64, rho: f64) -> f64 {
    let x = u / rho;
    let big_n = (p + 1) as f64;
    x.powi(p as i32 + 1) * ((2.0 * big_n + 1.0) / (1.0 - x) + 2.0 * x / (1.0 - x).powi(2))
}

/// The §3.11 input-truncation bound on the M2L output coefficients of degree j, for a
/// unit charge at scaled distance `u` from the source centre, without the factor
/// √((j − |i|)! (j + |i|)!): min over ρ in (|u|, |b|) of σʲ⁺¹ T(ρ) / (|b| − ρ)ʲ⁺¹.
/// The minimum is taken over 200 equally spaced interior values of ρ; any ρ in the
/// interval gives a valid bound. Returns one value per degree j ≤ p.
fn truncation_bounds(p: usize, u: f64, b: f64, sigma: f64) -> Vec<f64> {
    (0..=p)
        .map(|j| {
            if u == 0.0 {
                return 0.0;
            }
            (1..=200)
                .map(|k| {
                    let rho = u + (b - u) * k as f64 / 201.0;
                    sigma.powi(j as i32 + 1) * tail(p, u, rho) / (b - rho).powi(j as i32 + 1)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Rounding floor of the coefficient test, relative to the M2L term magnitudes.
const COEFFICIENT_FLOOR: f64 = 1e-13;

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// Compares M2L of P2M of a unit charge at `y` with P2L of the same charge in the
/// target frame, coefficient by coefficient, and returns the worst ratio of error to
/// bound plus floor. Panics if an error exceeds its bound plus floor.
fn check_coefficients_against_p2l(
    p: usize,
    source: &Frame<f64>,
    target: &Frame<f64>,
    y: [f64; 3],
    ws: &mut Workspace<f64>,
) -> f64 {
    let mut multipole = vec![0.0; len(p)];
    leaf::p2m(p, source, &[y], &[1.0], ws, &mut multipole);
    let mut got = vec![0.0; len(p)];
    direct::m2l(p, source, target, ws, &multipole, &mut got);
    let mut want = vec![0.0; len(p)];
    leaf::p2l(p, target, &[y], &[1.0], ws, &mut want);

    let (b, sigma) = l2l_shift(source, target);
    let u = source.scaled(y);
    let bounds = truncation_bounds(p, norm(u), norm(b), sigma);
    let scale = m2l_scale(&p2m_magnitudes(p, &[u], &[1.0]), sigma, norm(b));
    let layout = nd_fmm_math::Layout::new(p);
    let mut attained: f64 = 0.0;
    for j in 0..=p {
        for i in 0..=j as isize {
            let re = layout.idx(j, i);
            let mut error = (got[re] - want[re]).powi(2);
            if i > 0 {
                let im = layout.idx(j, -i);
                error += (got[im] - want[im]).powi(2);
            }
            let error = error.sqrt();
            let a = i.unsigned_abs();
            let size = (factorial(j - a) * factorial(j + a)).sqrt();
            let limit = size * bounds[j];
            let floor = COEFFICIENT_FLOOR * size * scale[j];
            assert!(
                error <= limit + floor,
                "{source:?} -> {target:?}, u = {u:?}, p = {p}, (j, i) = ({j}, {i}): error \
                 {error:.3e} exceeds bound {limit:.3e} + floor {floor:.3e}"
            );
            attained = attained.max(error / (limit + floor));
        }
    }
    attained
}

#[test]
fn m2l_coefficients_match_p2l_within_the_truncation_bound() {
    // Error measure: each complex output coefficient L̃′ⱼⁱ (i ≥ 0, modulus from the
    // slots ±i) of M2L of P2M of a unit charge, against P2L of that charge in the
    // target frame. It must lie within the per-coefficient input-truncation bound of
    // CONVENTIONS §3.11, σʲ⁺¹ √((j − |i|)! (j + |i|)!) T(ρ) / (|b| − ρ)ʲ⁺¹ minimised
    // over ρ in (|u|, |b|), plus a rounding floor of 1e-13 times the M2L term
    // magnitude of that coefficient, √((j − |i|)! (j + |i|)!) sⱼ with sⱼ the degree-j
    // term magnitude of `common::m2l_scale` (the unweighted size of a coefficient
    // whose weighted degree norm is sⱼ).
    //
    // Unlike the potential check, this exercises every order of Iₙ₊ⱼ up to degree 2p
    // and is tight for charges near the source centre, where the bound falls far below
    // the size of each term. Charges: one uniform in the source box and one uniform
    // in the box shrunk to a quarter, per offset. Frames: same level, all 316 V-list
    // offsets at p = 20, and every fifth offset at p in {0, 1, 4, 8, 12, 16}.
    let mut rng = SplitMix64::new(0x750e);
    let mut ws = Workspace::new(P_MAX_M2L);
    let attained = Worst::new("M2L coefficients vs P2L, error / (bound + floor)");
    let source = Frame::new([0.3, -1.1, 0.7], 0.37);
    for (index, d) in offsets().into_iter().enumerate() {
        let target = Frame::new(place(&source, d.map(|c| 2.0 * c)), source.radius);
        let charges = [
            place(&source, rng.in_cube()),
            place(&source, rng.in_cube().map(|c| 0.25 * c)),
        ];
        let mut degrees = vec![P_MAX_M2L];
        if index.is_multiple_of(5) {
            degrees.extend([0, 1, 4, 8, 12, 16]);
        }
        for p in degrees {
            for y in charges {
                attained.update(check_coefficients_against_p2l(
                    p, &source, &target, y, &mut ws,
                ));
            }
        }
    }
}

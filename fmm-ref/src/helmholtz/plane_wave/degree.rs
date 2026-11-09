//! The degree-matched quadrature (Helmholtz CH6.1f, `docs/helmholtz/tasks/CH6.1f.md`): a λ
//! rule fitted not to the kernel but to the exponential M2L's chain at a degree p, so that
//! the chain reproduces RCR at degree p.
//!
//! # Why
//!
//! The chain of CONVENTIONS §3.15 ("Exponential M2L") is exact per degree: with the exact
//! integral in place of the quadrature it is the M2L at degree p. A quadrature fitted to the
//! kernel e^{iκ|u|}/|u| integrates the waves of the full series, but the waves of a degree-p
//! multipole are a truncated series, which departs from the kernel's integrand at large λ
//! (the degree's "tail"); there a quadrature fitted to the kernel errs, the more the larger
//! the λ its nodes reach (CH6.1c's degree study). This rule is fitted to the truncated
//! integrand itself.
//!
//! # The integrand
//!
//! With q = κŝ = (t cos α, t sin α, iλ) (q·q = κ²), a source at s (box widths, from its
//! box's centre), a target at y (from its box's centre) and the centres' offset d, the
//! chain at degree p is the quadrature of S_p(q; s) e^{i q·d} T_p(q; y), where by
//! Gegenbauer's expansion, a polynomial identity that holds at complex q,
//!
//! S_p(q; s) = Σ_{n≤p} (2n + 1)(−i)ⁿ jₙ(κ|s|) Pₙ(q·s / (κ|s|)),
//! T_p(q; y) = Σ_{n≤p} (2n + 1) iⁿ jₙ(κ|y|) Pₙ(q·y / (κ|y|)),
//!
//! the truncations of e^{−i q·s} and e^{i q·y}. Each is a trigonometric polynomial of
//! degree p in α, finite as κ → 0: with z = κ|x|, a power scale σ = min(|z|, 1) and
//! ĵₙ = e^{−Im z} jₙ(z)/σⁿ (`nd_fmm_math::radial::bessel_j`), jₙ(z) Pₙ(a/z) =
//! e^{Im z} ĵₙ(z) Qₙ with Qₙ = σⁿ Pₙ(a/z), which Legendre's recurrence gives as
//! (n + 1) Qₙ₊₁ = (2n + 1)(a σ/z) Qₙ − n σ² Qₙ₋₁ (|σ/z| ≤ 1).
//!
//! # The rule
//!
//! 1. **Samples**: sources and targets at their boxes' corners (±½)³, face centres and
//!    centres (the corners and centres alone leave errors of up to 8ε between them at
//!    p = 24), and the 98 centre offsets d = (d_x, d_y, d_z), d_z ∈ {2, 3}, |d_x|, |d_y| ≤ 3,
//!    of the region of the kernel's quadrature (1 ≤ u_z ≤ 4, |u_x|, |u_y| ≤ 4); for the
//!    check, four seeded sources and targets in their boxes at the same offsets.
//! 2. **The trusted values**: RCR at degree p (`crate::helmholtz::{leaf, rcr}`: P2M of a unit
//!    charge, M2L, L2P, on a level of box width 1 and wavenumber κ) at every sample.
//! 3. **The full rule**: the generator's composite Gauss–Legendre rule on the contour of
//!    real t ([`super::rule`]), the panels doubled up to [`MAX_PANELS`] and the contour
//!    lengthened from Re λ = ln(1/τ) + 4 by p/4, p/2 and p (τ = ε/4 the rule's target),
//!    until its α-resolved sums of the integrand meet τ/4 of RCR's values on both sample
//!    sets.
//! 4. **Compression**: the generator's column-pivoted Gram–Schmidt with refitted weights,
//!    the right-hand side RCR's values, the tolerance divided by 4 until τ holds on both
//!    sets.
//! 5. **Counts**: every node keeps the orders m ≤ p (Fₖ = p; the chain keeps min(Fₖ, p));
//!    Nₖ is the fewest angles whose trapezoid of the node's integrand is
//!    within |wₖ|⁻¹ ε/(4s) of its α-resolved sum at every fit sample, by bisection (the
//!    chain is that trapezoid at any count: its conversions are exact on the expansions'
//!    orders, CONVENTIONS §3.15). If the
//!    result misses ε on the samples, the share is halved, down to 1/64 of its start.
//!
//! The error is absolute, in box widths, per unit charge, against RCR at degree p (not
//! against the kernel: [`Quadrature::kernel_error`] then measures the degree's own
//! truncation as well). [`Quadrature::shortfall`] says where it stops short of ε on the
//! samples. Everything is f64 and a function of κ, ε and p only (proposal §2, constraint
//! 9). The rule is slower than [`generate`](super::generate), seconds per κ in release.

use core::f64::consts::{SQRT_2, TAU};

use nd_fmm_math::Layout;
use nd_fmm_math::radial::bessel_j;

use super::rule::{Rule, compress, panel_rule_to};
use super::{Bound, C, FLOOR, MAX_PANELS, Quadrature, Shortfall};
use crate::Frame;
use crate::helmholtz::{Level, Workspace, leaf, rcr};

/// The largest lateral offset of two centres of the samples, 3√2 box widths.
const OFFSET_REACH: f64 = 3.0 * SQRT_2;

/// Sources, targets (from their boxes' centres) and centre offsets, box widths; the
/// sample (d, s, y) is row (d · sources + s) · targets + y.
struct Samples {
    sources: Vec<[f64; 3]>,
    targets: Vec<[f64; 3]>,
    offsets: Vec<[f64; 3]>,
}

impl Samples {
    /// The centre offsets of the region.
    fn offsets() -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(98);
        for dz in [2.0, 3.0] {
            for dy in -3..=3 {
                for dx in -3..=3 {
                    out.push([dx as f64, dy as f64, dz]);
                }
            }
        }
        out
    }

    /// The fit's samples: the corners, the face centres and the centre of both boxes.
    fn fit() -> Self {
        let mut points: Vec<[f64; 3]> = (0..8)
            .map(|c| [c & 4, c & 2, c & 1].map(|b| if b == 0 { -0.5 } else { 0.5 }))
            .collect();
        points.push([0.0; 3]);
        for axis in 0..3 {
            for side in [-0.5, 0.5] {
                let mut face = [0.0; 3];
                face[axis] = side;
                points.push(face);
            }
        }
        Self {
            sources: points.clone(),
            targets: points,
            offsets: Self::offsets(),
        }
    }

    /// The check's samples: four seeded points in each box.
    fn check() -> Self {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut point = || [(); 3].map(|_| next());
        let sources = (0..4).map(|_| point()).collect();
        let targets = (0..4).map(|_| point()).collect();
        Self {
            sources,
            targets,
            offsets: Self::offsets(),
        }
    }

    fn rows(&self) -> usize {
        self.offsets.len() * self.sources.len() * self.targets.len()
    }
}

/// RCR at degree p at every sample: P2M of a unit charge, M2L, L2P, on the level of box
/// width 1 and wavenumber κ.
fn rcr_values(kappa: C, p: usize, samples: &Samples) -> Vec<C> {
    let level = Level::new(kappa, 0.5);
    let mut ws = Workspace::new(p);
    let len = Layout::new(p).len();
    let origin = Frame::new([0.0; 3], 0.5);
    let multipoles: Vec<Vec<f64>> = samples
        .sources
        .iter()
        .map(|s| {
            let mut m = vec![0.0; 2 * len];
            leaf::p2m(
                p,
                &level,
                &origin,
                &[*s],
                &[C::new(1.0, 0.0)],
                &mut ws,
                &mut m,
            );
            m
        })
        .collect();
    let mut out = Vec::with_capacity(samples.rows());
    let mut local = vec![0.0; 2 * len];
    let mut values = vec![C::new(0.0, 0.0); samples.targets.len()];
    for d in &samples.offsets {
        let to = Frame::new(*d, 0.5);
        let table = rcr::m2l_table(p, p, &level, &origin, &to);
        let points: Vec<[f64; 3]> = samples
            .targets
            .iter()
            .map(|y| core::array::from_fn(|a| d[a] + y[a]))
            .collect();
        for m in &multipoles {
            local.fill(0.0);
            rcr::m2l(p, p, &level, &origin, &to, &table, &mut ws, m, &mut local);
            values.fill(C::new(0.0, 0.0));
            leaf::l2p(p, &level, &to, &local, &points, &mut ws, &mut values, None);
            out.extend_from_slice(&values);
        }
    }
    out
}

/// One point's truncated expansion (the [module documentation](self)): its x, and
/// (2n + 1) (±i)ⁿ e^{Im z} ĵₙ(z) for n ≤ p, the scale s = min(|z|, 1) of ĵₙ, and s/z;
/// [`Expansion::new`] gives `None` at the centre, where the expansion is 1.
struct Expansion {
    x: [f64; 3],
    radial: Vec<C>,
    scale: f64,
    over_z: C,
}

impl Expansion {
    fn new(kappa: C, p: usize, x: [f64; 3], sign: f64) -> Option<Self> {
        let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        if r == 0.0 {
            return None;
        }
        let z = kappa * r;
        let scale = z.norm().min(1.0);
        let mut radial = vec![C::new(0.0, 0.0); p + 1];
        bessel_j(z, scale, &mut radial, None);
        let (damping, sigma) = (z.im.exp(), C::new(0.0, sign));
        let mut power = C::new(1.0, 0.0);
        for (n, v) in radial.iter_mut().enumerate() {
            *v *= power * damping * (2 * n + 1) as f64;
            power *= sigma;
        }
        Some(Self {
            x,
            radial,
            scale,
            over_z: C::new(scale, 0.0) / z,
        })
    }

    /// The truncated expansion at q = (t cos α, t sin α, iλ).
    fn at(this: &Option<Self>, q: [C; 3]) -> C {
        let Some(e) = this else {
            return C::new(1.0, 0.0);
        };
        let a = q[0] * e.x[0] + q[1] * e.x[1] + q[2] * e.x[2];
        let (ac, s2) = (a * e.over_z, e.scale * e.scale);
        let (mut prev, mut q_n) = (C::new(0.0, 0.0), C::new(1.0, 0.0));
        let mut sum = e.radial[0];
        for n in 0..e.radial.len() - 1 {
            let next = (ac * q_n * (2 * n + 1) as f64 - prev * (n as f64 * s2)) / (n + 1) as f64;
            prev = q_n;
            q_n = next;
            sum += e.radial[n + 1] * q_n;
        }
        sum
    }
}

/// The samples' expansions: the sources' S_p and the targets' T_p.
struct Expansions<'a> {
    samples: &'a Samples,
    sources: Vec<Option<Expansion>>,
    targets: Vec<Option<Expansion>>,
}

impl<'a> Expansions<'a> {
    fn new(kappa: C, p: usize, samples: &'a Samples) -> Self {
        Self {
            samples,
            sources: samples
                .sources
                .iter()
                .map(|&s| Expansion::new(kappa, p, s, -1.0))
                .collect(),
            targets: samples
                .targets
                .iter()
                .map(|&y| Expansion::new(kappa, p, y, 1.0))
                .collect(),
        }
    }

    /// The node's integrand at every sample, its trapezoid over `n` angles:
    /// (1/n) Σⱼ S_p e^{i q·d} T_p (the weight not applied).
    fn column(&self, t: f64, lambda: C, n: usize) -> Vec<C> {
        let samples = self.samples;
        let (ns, nt) = (samples.sources.len(), samples.targets.len());
        let mut col = vec![C::new(0.0, 0.0); samples.rows()];
        let mut st = vec![C::new(0.0, 0.0); ns * nt];
        for j in 0..n {
            let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
            let q = [C::new(t * ca, 0.0), C::new(t * sa, 0.0), C::i() * lambda];
            let tv: Vec<C> = self.targets.iter().map(|e| Expansion::at(e, q)).collect();
            for (i, e) in self.sources.iter().enumerate() {
                let sv = Expansion::at(e, q);
                for (o, v) in st[i * nt..][..nt].iter_mut().zip(&tv) {
                    *o = sv * v;
                }
            }
            for (d, row) in samples.offsets.iter().zip(col.chunks_exact_mut(ns * nt)) {
                let e = (-lambda * d[2] + C::i() * t * (d[0] * ca + d[1] * sa)).exp();
                for (o, v) in row.iter_mut().zip(&st) {
                    *o += e * v;
                }
            }
        }
        let scale = 1.0 / n as f64;
        for v in &mut col {
            *v *= scale;
        }
        col
    }

    /// Every node's resolved column.
    fn columns(&self, rule: &Rule, p: usize) -> Vec<Vec<C>> {
        rule.t
            .iter()
            .zip(&rule.lambda)
            .map(|(&t, &l)| self.column(t, l, resolved_count(t, p)))
            .collect()
    }
}

/// Angles that resolve a node's integrand: the truncated expansions have orders up to p
/// each, the shift's orders reach about t ρ (ρ ≤ [`OFFSET_REACH`]) with a tail of a few
/// (t ρ)^{1/3}.
fn resolved_count(t: f64, p: usize) -> usize {
    let x = t * OFFSET_REACH;
    2 * p + (x + 15.0 * x.cbrt()).ceil() as usize + 30
}

/// The largest |Σₖ wₖ colₖ − v| over the rows.
fn error(cols: &[Vec<C>], weight: &[C], values: &[C]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let q: C = cols.iter().zip(weight).map(|(c, w)| c[i] * w).sum();
            (q - v).norm()
        })
        .fold(0.0, f64::max)
}

/// The degree-matched quadrature for κ at ε and degree p ([module
/// documentation](super), "Matched to a degree"): the λ rule fitted to the exponential
/// M2L's chain at degree p, the truncations at degree p of the source's and the target's
/// plane-wave expansions (Gegenbauer's series), against RCR at degree p (this crate's
/// `helmholtz::{leaf, rcr}`) at sources and targets on their boxes' corners, face centres
/// and centres and the 98 centre offsets of the region (d_z ∈ {2, 3}, |d_x|, |d_y| ≤ 3),
/// checked at seeded points. The contour runs from Re λ = ln(4/ε) + 4 (the kernel's at ε/4)
/// up to p further until the full rule meets RCR; every node keeps the orders m ≤ p
/// (Fₖ = p), and Nₖ is the count, found by bisection, within the share ε/(4s) of the node's
/// resolved trapezoid (the final check against RCR catches a count the bisection misjudges).
///
/// The chain at degree p is then within ε of RCR at degree p on the samples, absolute in
/// box widths per unit charge, or [`Quadrature::shortfall`] says how far it is. A function
/// of κ, ε and p only (proposal §2, constraint 9); seconds to a minute in release.
///
/// # Panics
///
/// If Re κ or Im κ is negative or not finite, κ is zero, or ε is not positive and finite.
pub fn generate_for_degree(kappa: C, eps: f64, p: usize) -> Quadrature {
    assert!(
        kappa.re.is_finite() && kappa.im.is_finite() && kappa.re >= 0.0 && kappa.im >= 0.0,
        "κ = {kappa} outside the closed first quadrant"
    );
    assert!(kappa.norm() > 0.0, "κ = 0 is Laplace");
    assert!(
        eps.is_finite() && eps > 0.0,
        "ε = {eps:e} is not positive and finite"
    );
    let requested = eps;
    let eps = eps.max(FLOOR);
    let target = eps / 4.0;
    let (fit, check) = (Samples::fit(), Samples::check());
    let (fit_values, check_values) = (rcr_values(kappa, p, &fit), rcr_values(kappa, p, &check));
    let (fit_x, check_x) = (
        Expansions::new(kappa, p, &fit),
        Expansions::new(kappa, p, &check),
    );

    // The full rule (step 3).
    let base = (1.0 / target).ln() + 4.0;
    let mut full = None;
    for extra in [0.0, 0.25, 0.5, 1.0] {
        let lambda_max = base + extra * p as f64;
        let mut n = 1;
        loop {
            let rule = panel_rule_to(kappa, lambda_max, n);
            let cols = fit_x.columns(&rule, p);
            let check_cols = check_x.columns(&rule, p);
            let e = error(&cols, &rule.weight, &fit_values).max(error(
                &check_cols,
                &rule.weight,
                &check_values,
            ));
            let met = e <= target / 4.0;
            if met || n >= MAX_PANELS {
                full = Some((rule, cols, check_cols, !met));
                break;
            }
            n *= 2;
        }
        if full.as_ref().is_some_and(|f| !f.3) {
            break;
        }
    }
    let (rule, cols, check_cols, panels_capped) = full.expect("at least one contour");

    // Compression (step 4).
    let mut tol = target;
    let (chosen, weight, compression_capped) = loop {
        let (chosen, w) = compress(&cols, &fit_values, tol);
        let pick = |c: &[Vec<C>]| -> Vec<Vec<C>> { chosen.iter().map(|&j| c[j].clone()).collect() };
        let e =
            error(&pick(&cols), &w, &fit_values).max(error(&pick(&check_cols), &w, &check_values));
        if e <= target || tol < target * 1e-4 {
            break (chosen, w, e > target);
        }
        tol /= 4.0;
    };
    let lambda: Vec<C> = chosen.iter().map(|&j| rule.lambda[j]).collect();
    let t: Vec<f64> = chosen.iter().map(|&j| rule.t[j]).collect();
    let s = chosen.len();

    // Counts (step 5).
    let mut share = eps / (4.0 * s as f64);
    let mut attempts = 0;
    loop {
        let alpha: Vec<usize> = (0..s)
            .map(|k| {
                let resolved = &cols[chosen[k]];
                let bound = share / weight[k].norm();
                let fits = |n: usize| {
                    fit_x
                        .column(t[k], lambda[k], n)
                        .iter()
                        .zip(resolved)
                        .all(|(a, b)| (a - b).norm() <= bound)
                };
                let (mut lo, mut hi) = (1, resolved_count(t[k], p));
                if fits(lo) {
                    return lo;
                }
                while hi - lo > 1 {
                    let mid = (lo + hi) / 2;
                    if fits(mid) { hi = mid } else { lo = mid }
                }
                hi
            })
            .collect();
        let at = |x: &Expansions<'_>, values: &[C]| {
            let cols: Vec<Vec<C>> = (0..s)
                .map(|k| x.column(t[k], lambda[k], alpha[k]))
                .collect();
            error(&cols, &weight, values)
        };
        let e = at(&fit_x, &fit_values).max(at(&check_x, &check_values));
        attempts += 1;
        if e <= eps || attempts > 6 {
            let mut q = Quadrature::new(kappa, lambda, weight, alpha, vec![p; s]);
            q.eps = requested;
            q.degree = Some(p);
            if e > requested {
                let mut bounds = Vec::new();
                if requested < FLOOR {
                    bounds.push(Bound::Floor);
                }
                if panels_capped {
                    bounds.push(Bound::Panels);
                }
                if compression_capped {
                    bounds.push(Bound::Compression);
                }
                if e > eps {
                    bounds.push(Bound::Tightening);
                }
                q.shortfall = Some(Shortfall { error: e, bounds });
            }
            return q;
        }
        share /= 2.0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::kernel;
    use super::*;

    /// At a degree that resolves it, the truncated expansion is the plane wave
    /// e^{∓i q·x}, on and off the contour (q·q = κ²), for small, real and complex κ:
    /// relative to the largest of 1 and the plane wave, within 10⁻¹².
    #[test]
    fn the_expansions_are_the_plane_waves() {
        let x = [0.3, -0.4, 0.45];
        for kappa in [C::new(0.004, 0.0), C::new(1.25, 0.0), C::new(2.0, 1.0)] {
            for (t, alpha) in [(0.5, 0.3), (3.0, 1.1), (6.0, 4.0)] {
                let lambda = (C::new(t * t, 0.0) - kappa * kappa).sqrt();
                let (sa, ca) = f64::sin_cos(alpha);
                let q = [C::new(t * ca, 0.0), C::new(t * sa, 0.0), C::i() * lambda];
                let qx = q[0] * x[0] + q[1] * x[1] + q[2] * x[2];
                for sign in [-1.0, 1.0] {
                    let e = Expansion::new(kappa, 60, x, sign);
                    let exact = (C::new(0.0, sign) * qx).exp();
                    let got = Expansion::at(&e, q);
                    let scale = exact.norm().max(1.0);
                    assert!(
                        (got - exact).norm() <= 1e-12 * scale,
                        "κ {kappa}, t {t}, sign {sign}: {got} against {exact}"
                    );
                }
            }
        }
        assert_eq!(
            Expansion::at(
                &Expansion::new(C::new(1.0, 0.0), 8, [0.0; 3], 1.0),
                [C::i(); 3]
            ),
            C::new(1.0, 0.0)
        );
    }

    /// RCR at a high degree is the kernel e^{iκ|u|}/|u| of the samples, u = d + y − s:
    /// the trusted values' units and normalisation (within 10⁻¹⁰ at p = 40, at points
    /// within 0.4 of their centres, where the degree converges; at the corners p = 30 still
    /// truncates at about 5·10⁻⁷).
    #[test]
    fn rcr_values_are_the_kernel_at_a_high_degree() {
        let kappa = C::new(1.25, 0.25);
        let samples = Samples {
            sources: vec![[0.2, 0.2, 0.2], [0.0; 3]],
            targets: vec![[-0.2, -0.2, -0.2], [0.1, 0.2, -0.3]],
            offsets: vec![[0.0, 0.0, 2.0], [3.0, -3.0, 2.0]],
        };
        let values = rcr_values(kappa, 40, &samples);
        let mut row = 0;
        for d in &samples.offsets {
            for s in &samples.sources {
                for y in &samples.targets {
                    let u = core::array::from_fn(|a| d[a] + y[a] - s[a]);
                    let e = (values[row] - kernel(kappa, u)).norm();
                    assert!(e <= 1e-10, "row {row}: {e:.2e}");
                    row += 1;
                }
            }
        }
    }
}

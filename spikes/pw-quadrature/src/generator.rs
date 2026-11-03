//! An independent generator of the quadratures, for any κ = k w with Re κ, Im κ ≥ 0.
//!
//! 1. **The contour of real t.** λ = −i q with q = √(κ² − t²), Im q ≥ 0 (the outgoing
//!    branch), and dλ = t dt / λ, for t from 0 to t_max, where Re λ reaches
//!    ln(1/ε) + 4 (the tail beyond is below ε/50 at z ≥ 1). On it J₀(t ρ) has a real
//!    argument, so the waves never grow (|e^{i t (x cos α + y sin α)}| = 1) and
//!    |e^{−λ z}| ≤ 1. For Re κ > 0 the variable is θ on [0, π/2] (t = Re κ sin θ) and
//!    φ on [0, φ_max] (t = Re κ cosh φ), which removes the 1/√ singularity of t/λ at
//!    t = κ for real κ; for 0 < Im κ ≪ Re κ the panels are graded geometrically towards
//!    θ = π/2 and φ = 0, where the near-singularity sits at a width √(Im κ / Re κ).
//!    For Re κ = 0 the variable is t itself, graded towards 0 at a width Im κ.
//! 2. **Panels.** Composite Gauss–Legendre of [`ORDER`] points, the panel count doubled
//!    until the λ sum meets ε/4 on the (z, ρ) grid against the kernel.
//! 3. **Compression.** A column-pivoted Gram–Schmidt of the matrix of the integrand at
//!    the (z, ρ) samples selects nodes until the residual is below the tolerance; the
//!    weights of the selected nodes are refitted by least squares to the full rule's
//!    values (the first step of a generalised Gaussian quadrature, without its
//!    Gauss–Newton elimination). The tolerance is tightened until ε/4 holds again.
//! 4. **α and Fourier counts.** Per node the fewest α nodes Nₖ with
//!    2 |wₖ| e^{−Re λₖ} max_{m ≥ N, x ≤ tₖ ρ_max} |J_m(x)| ≤ ε / (4s) (the trapezoid's
//!    error in α), and the fewest orders Fₖ with the same bound on the J_m, m > Fₖ, at
//!    x ≤ tₖ r_s,
//!    r_s = √2/2 the lateral reach of a source in its box.
//!
//! The result is checked against the kernel on the region with its α counts; if it
//! misses ε the shares of step 4 are halved.

use crate::bessel::j_all;
use crate::gauss::legendre;
use crate::quadrature::Quadrature;
use crate::region::Grid;
use crate::{C, kernel};

/// Gauss–Legendre points per panel.
pub const ORDER: usize = 20;

/// The smallest ε the generator aims for: f64 round-off of sums of 10³ terms of size 1.
pub const FLOOR: f64 = 1e-13;

/// The most panels per segment before the λ rule stops refining.
pub const MAX_PANELS: usize = 64;

/// ρ_max = 4√2, the region's largest lateral offset.
pub const RHO_MAX: f64 = 4.0 * core::f64::consts::SQRT_2;

/// r_s = √2/2, the largest lateral offset of a point from its box's centre.
pub const SOURCE_REACH: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// A λ rule on the contour of real t: real t, complex λ and weights.
#[derive(Clone, Debug, Default)]
pub struct Rule {
    /// The real transverse wavenumbers tₖ.
    pub t: Vec<f64>,
    /// λₖ.
    pub lambda: Vec<C>,
    /// wₖ (for dλ).
    pub weight: Vec<C>,
}

/// λ = −i q, q = √(q²) with Im q ≥ 0.
fn lambda_of(q2: C) -> C {
    let mut q = q2.sqrt();
    if q.im < 0.0 {
        q = -q;
    }
    -C::i() * q
}

/// Panel breakpoints on [0, b]: `n` uniform panels, the one at `towards` (0 or b)
/// subdivided geometrically by halves down to `width` / 4.
fn panels(b: f64, n: usize, towards_end: bool, width: f64) -> Vec<(f64, f64)> {
    let h = b / n as f64;
    let mut out: Vec<(f64, f64)> = (0..n).map(|i| (i as f64 * h, (i + 1) as f64 * h)).collect();
    if width > 0.0 && width < h {
        let (lo, hi) = if towards_end {
            out.pop().unwrap()
        } else {
            out.remove(0)
        };
        let mut pieces = Vec::new();
        let (mut a, mut c) = (lo, hi);
        while c - a > width / 4.0 {
            let m = 0.5 * (a + c);
            if towards_end {
                pieces.push((a, m));
                a = m;
            } else {
                pieces.push((m, c));
                c = m;
            }
        }
        pieces.push((a, c));
        if towards_end {
            out.extend(pieces);
        } else {
            pieces.reverse();
            let mut v = pieces;
            v.extend(out);
            out = v;
        }
    }
    out
}

/// The composite rule with `n` panels per segment (step 1 and 2 of the module
/// documentation, one panel count).
pub fn panel_rule(kappa: C, eps: f64, n: usize) -> Rule {
    let (kr, ki) = (kappa.re, kappa.im);
    assert!(
        kr >= 0.0 && ki >= 0.0,
        "κ = {kappa} outside the first quadrant"
    );
    let lambda_max = (1.0 / eps).ln() + 4.0;
    let t_max = (lambda_max * lambda_max + kappa.norm_sqr()).sqrt();
    let (gx, gw) = legendre(ORDER, 0.0, 1.0);
    let mut rule = Rule::default();
    let mut push = |a: f64, b: f64, f: &dyn Fn(f64) -> (f64, C, f64)| {
        for (x, w) in gx.iter().zip(&gw) {
            let v = a + (b - a) * x;
            // (t, q², dt/dv) at v.
            let (t, q2, dt) = f(v);
            let lambda = lambda_of(q2);
            rule.t.push(t);
            rule.lambda.push(lambda);
            rule.weight.push(w * (b - a) * t * dt / lambda);
        }
    };
    if kr > 0.0 {
        let delta = if ki > 0.0 { (ki / kr).sqrt() } else { 0.0 };
        let theta = |v: f64| {
            let (s, c) = v.sin_cos();
            (
                kr * s,
                C::new((kr * c).powi(2) - ki * ki, 2.0 * kr * ki),
                kr * c,
            )
        };
        for (a, b) in panels(core::f64::consts::FRAC_PI_2, n, true, delta) {
            push(a, b, &theta);
        }
        let phi_max = (t_max / kr).acosh();
        let phi = |v: f64| {
            let sh = v.sinh();
            (
                kr * v.cosh(),
                C::new(-(kr * sh).powi(2) - ki * ki, 2.0 * kr * ki),
                kr * sh,
            )
        };
        for (a, b) in panels(phi_max, 2 * n, false, delta) {
            push(a, b, &phi);
        }
    } else {
        let plain = |v: f64| (v, C::new(-ki * ki - v * v, 0.0), 1.0);
        for (a, b) in panels(t_max, 2 * n, false, ki) {
            push(a, b, &plain);
        }
    }
    rule
}

/// The (z, ρ) samples of the λ sums: the fit's, z ∈ [1, 4] in 25 steps of 1/8 and
/// ρ ∈ [0, ρ_max] in 65; or (`midpoints`) the check's, at the midpoints between them.
fn samples(midpoints: bool) -> Vec<(f64, f64)> {
    let (nz, nr) = if midpoints { (24, 64) } else { (25, 65) };
    let off = if midpoints { 0.5 } else { 0.0 };
    let rho: Vec<f64> = (0..nr).map(|i| RHO_MAX * (i as f64 + off) / 64.0).collect();
    (0..nz)
        .flat_map(|iz| {
            let z = 1.0 + (iz as f64 + off) / 8.0;
            rho.iter().map(move |&r| (z, r))
        })
        .collect()
}

/// The matrix of the integrand e^{−λₖ z} J₀(tₖ ρ) at the samples, column by node.
fn columns(rule: &Rule, samples: &[(f64, f64)]) -> Vec<Vec<C>> {
    let mut rhos: Vec<f64> = samples.iter().map(|s| s.1).collect();
    rhos.sort_by(f64::total_cmp);
    rhos.dedup();
    rule.t
        .iter()
        .zip(&rule.lambda)
        .map(|(&t, &l)| {
            let j0: Vec<f64> = rhos.iter().map(|&r| j_all(t * r, 0)[0]).collect();
            samples
                .iter()
                .map(|&(z, r)| {
                    let i = rhos.partition_point(|&v| v < r);
                    (-l * z).exp() * j0[i]
                })
                .collect()
        })
        .collect()
}

/// The largest |Σₖ wₖ e^{−λₖ z} J₀(tₖ ρ) − G| over the samples.
fn lambda_error(kappa: C, cols: &[Vec<C>], weight: &[C], samples: &[(f64, f64)]) -> f64 {
    samples
        .iter()
        .enumerate()
        .map(|(i, &(z, r))| {
            let q: C = cols.iter().zip(weight).map(|(c, w)| c[i] * w).sum();
            (q - kernel(kappa, [r, 0.0, z])).norm()
        })
        .fold(0.0, f64::max)
}

/// Column-pivoted modified Gram–Schmidt: the selected columns, in order, until the
/// largest remaining column norm is at most `tol` times the largest initial one; then
/// the least-squares weights of the selected columns for the right-hand side `rhs`.
fn compress(cols: &[Vec<C>], rhs: &[C], tol: f64) -> (Vec<usize>, Vec<C>) {
    let m = rhs.len();
    let mut work: Vec<Vec<C>> = cols.to_vec();
    let norm2 = |v: &[C]| v.iter().map(|x| x.norm_sqr()).sum::<f64>();
    let mut norms: Vec<f64> = work.iter().map(|c| norm2(c)).collect();
    let top = norms.iter().cloned().fold(0.0, f64::max).sqrt();
    let mut chosen = Vec::new();
    let mut q: Vec<Vec<C>> = Vec::new();
    let mut r: Vec<Vec<C>> = Vec::new(); // r[j][i], i ≤ j: the R factor by column
    loop {
        let (best, &bn) = norms
            .iter()
            .enumerate()
            .filter(|(j, _)| !chosen.contains(j))
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        if chosen.len() == cols.len() || bn.sqrt() <= tol * top || chosen.len() == m {
            break;
        }
        // Re-orthogonalise the chosen column against q twice (MGS², for stability).
        let mut v = work[best].clone();
        for _ in 0..2 {
            for qi in &q {
                let d: C = qi.iter().zip(&v).map(|(a, b)| a.conj() * b).sum();
                for (x, a) in v.iter_mut().zip(qi) {
                    *x -= d * a;
                }
            }
        }
        // R's column for `best` in the original basis: the projections of cols[best].
        let orig = &cols[best];
        let mut rc: Vec<C> = q
            .iter()
            .map(|qi| qi.iter().zip(orig).map(|(a, b)| a.conj() * b).sum())
            .collect();
        let nv = norm2(&v).sqrt();
        for x in v.iter_mut() {
            *x /= nv;
        }
        rc.push(C::new(nv, 0.0));
        // Update the remaining columns' residual norms.
        for (j, w) in work.iter_mut().enumerate() {
            if chosen.contains(&j) || j == best {
                continue;
            }
            let d: C = v.iter().zip(w.iter()).map(|(a, b)| a.conj() * b).sum();
            for (x, a) in w.iter_mut().zip(&v) {
                *x -= d * a;
            }
            norms[j] = norm2(w);
        }
        chosen.push(best);
        q.push(v);
        r.push(rc);
        norms[best] = 0.0;
    }
    // Least squares: R w = Qᴴ rhs, back substitution.
    let k = chosen.len();
    let qb: Vec<C> = q
        .iter()
        .map(|qi| qi.iter().zip(rhs).map(|(a, b)| a.conj() * b).sum())
        .collect();
    let mut w = vec![C::new(0.0, 0.0); k];
    for i in (0..k).rev() {
        let mut s = qb[i];
        for j in i + 1..k {
            s -= r[j][i] * w[j];
        }
        w[i] = s / r[i][i];
    }
    (chosen, w)
}

/// The generated quadrature's λ rule before the α counts, meeting `target` on the fit's
/// and the check's samples: (rule, the panel count, the nodes before compression).
pub fn lambda_rule(kappa: C, eps: f64, target: f64) -> (Rule, usize, usize) {
    let fit = samples(false);
    let check = samples(true);
    let mut n = 1;
    let (full, cols, check_cols) = loop {
        let rule = panel_rule(kappa, eps.min(target), n);
        let cols = columns(&rule, &fit);
        let check_cols = columns(&rule, &check);
        let e = lambda_error(kappa, &cols, &rule.weight, &fit).max(lambda_error(
            kappa,
            &check_cols,
            &rule.weight,
            &check,
        ));
        if e <= target / 4.0 || n >= MAX_PANELS {
            break (rule, cols, check_cols);
        }
        n *= 2;
    };
    let rhs: Vec<C> = (0..fit.len())
        .map(|i| cols.iter().zip(&full.weight).map(|(c, w)| c[i] * w).sum())
        .collect();
    let mut tol = target;
    loop {
        let (chosen, w) = compress(&cols, &rhs, tol);
        let pick = |c: &[Vec<C>]| -> Vec<Vec<C>> { chosen.iter().map(|&j| c[j].clone()).collect() };
        let e = lambda_error(kappa, &pick(&cols), &w, &fit).max(lambda_error(
            kappa,
            &pick(&check_cols),
            &w,
            &check,
        ));
        if e <= target || tol < target * 1e-4 {
            let rule = Rule {
                t: chosen.iter().map(|&j| full.t[j]).collect(),
                lambda: chosen.iter().map(|&j| full.lambda[j]).collect(),
                weight: w,
            };
            return (rule, n, full.t.len());
        }
        tol /= 4.0;
    }
}

/// The fewest n ≥ 1 with `scale` max_{m ≥ n} max_{x' ≤ x} |J_m(x')| ≤ `share`: the
/// trapezoid's error in α for J₀ up to x is about 2|J_N|, and a source's orders above F
/// at reach x are the J_m with m > F. The maximum over all larger orders matters below
/// the turning point m ≈ x, where a single |J_m| can be small at a zero of J_m while
/// the next orders are not.
fn tail_count(x: f64, scale: f64, share: f64) -> usize {
    let top = (x.ceil() as usize) * 2 + 80;
    let mut worst = vec![0.0f64; top + 1];
    for i in 1..=64 {
        let j = j_all(x * i as f64 / 64.0, top);
        for (w, v) in worst.iter_mut().zip(&j) {
            *w = w.max(v.abs());
        }
    }
    for m in (0..top).rev() {
        worst[m] = worst[m].max(worst[m + 1]);
    }
    (1..top).find(|&n| scale * worst[n] <= share).unwrap_or(top)
}

/// The generated quadrature for κ at ε (the module documentation), with its α and
/// Fourier counts. Nₖ also covers the chain: at least the count for the targets' reach
/// from the source box's centre, ρ_max − r_s, plus Fₖ, so the truncated source orders
/// do not alias.
///
/// At its bounds ([`MAX_PANELS`], the compression's tolerance floor, a fixed number of
/// tightenings, ε below [`FLOOR`]) it returns its best quadrature even if that misses ε;
/// a caller that relies on ε checks [`Quadrature::kernel_errors`].
pub fn generate(kappa: C, eps: f64) -> Quadrature {
    let eps = eps.max(FLOOR);
    let grid = Grid::region();
    let mut target = eps / 4.0;
    'rule: loop {
        let (rule, _, _) = lambda_rule(kappa, eps, target);
        let s = rule.t.len();
        let mut share = eps / (4.0 * s as f64);
        loop {
            let mut physical = Vec::with_capacity(s);
            let mut fourier = Vec::with_capacity(s);
            for k in 0..s {
                let scale = 2.0 * rule.weight[k].norm() * (-rule.lambda[k].re).exp();
                let t = rule.t[k];
                let f = tail_count(t * SOURCE_REACH, scale, share) - 1;
                let n = tail_count(t * RHO_MAX, scale, share)
                    .max(tail_count(t * (RHO_MAX - SOURCE_REACH), scale, share) + f);
                physical.push(n);
                fourier.push(f);
            }
            let q = Quadrature::new(
                kappa,
                rule.lambda.clone(),
                rule.weight.clone(),
                physical,
                fourier,
            );
            if q.kernel_errors(&grid).absolute <= eps || target < eps / 256.0 {
                return q;
            }
            if q.lambda_errors(&grid).absolute > eps / 2.0 {
                target /= 4.0;
                continue 'rule;
            }
            share /= 2.0;
            if share < eps / (4096.0 * s as f64) {
                return q;
            }
        }
    }
}

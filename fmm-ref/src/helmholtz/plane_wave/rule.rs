//! The λ rule of the generator: the contour of real t, its composite Gauss–Legendre
//! panels and their compression (steps 1 to 3 of the [module documentation](super)), in
//! S4's order of operations (`spikes/pw-quadrature/src/generator.rs`).

use super::special::{j_all, legendre};
use super::{C, ORDER, RHO_MAX, kernel};

/// A λ rule on the contour of real t: real t, complex λ and weights.
#[derive(Clone, Debug, Default)]
pub(super) struct Rule {
    /// The real transverse wavenumbers tₖ.
    pub(super) t: Vec<f64>,
    /// λₖ.
    pub(super) lambda: Vec<C>,
    /// wₖ (for dλ).
    pub(super) weight: Vec<C>,
}

/// What [`lambda_rule`] returns: the rule, and whether a bound stopped it short of its
/// target.
#[derive(Clone, Debug)]
pub(super) struct Fitted {
    /// The compressed rule.
    pub(super) rule: Rule,
    /// The panels per segment of the full rule.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "read by the panels' convergence test")
    )]
    pub(super) panels: usize,
    /// The panels reached their cap with the full rule's error above target/4.
    pub(super) panels_capped: bool,
    /// The compression reached its tolerance floor with the error above the target.
    pub(super) compression_capped: bool,
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
            out.pop().expect("at least one panel")
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

/// The composite rule with `n` panels per segment (steps 1 and 2 of the module
/// documentation, one panel count).
pub(super) fn panel_rule(kappa: C, eps: f64, n: usize) -> Rule {
    let (kr, ki) = (kappa.re, kappa.im);
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
    // r[j][i], i ≤ j: the R factor by column.
    let mut r: Vec<Vec<C>> = Vec::new();
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

/// The λ rule before the α counts, meeting `target` on the fit's and the check's
/// samples if it can: the panels doubled from one per segment up to `max_panels`, then
/// the compression's tolerance divided by 4 from `target` down to 10⁻⁴ `target`.
pub(super) fn lambda_rule(kappa: C, eps: f64, target: f64, max_panels: usize) -> Fitted {
    let fit = samples(false);
    let check = samples(true);
    let mut n = 1;
    let (full, cols, check_cols, panels, panels_capped) = loop {
        let rule = panel_rule(kappa, eps.min(target), n);
        let cols = columns(&rule, &fit);
        let check_cols = columns(&rule, &check);
        let e = lambda_error(kappa, &cols, &rule.weight, &fit).max(lambda_error(
            kappa,
            &check_cols,
            &rule.weight,
            &check,
        ));
        if e <= target / 4.0 || n >= max_panels {
            break (rule, cols, check_cols, n, e > target / 4.0);
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
            return Fitted {
                rule,
                panels,
                panels_capped,
                compression_capped: e > target,
            };
        }
        tol /= 4.0;
    }
}

/// The fewest n ≥ 1 with `scale` max_{m ≥ n} max_{x' ≤ x} |J_m(x')| ≤ `share`: the
/// trapezoid's error in α for J₀ up to x is about 2|J_N|, and a source's orders above F
/// at reach x are the J_m with m > F. The maximum over all larger orders matters below
/// the turning point m ≈ x, where a single |J_m| can be small at a zero of J_m while
/// the next orders are not.
pub(super) fn tail_count(x: f64, scale: f64, share: f64) -> usize {
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

#[cfg(test)]
mod tests {
    use super::super::MAX_PANELS;
    use super::*;

    /// The λ rule converges: at κ = 10, ε = 10⁻⁶ the full rule with twice the panels of
    /// the fitted count (the full rule of target ε/4) changes the sums Σₖ wₖ e^{−λₖ z} J₀(tₖ ρ) by at most ε/4 (absolute)
    /// at nine (z, ρ) of the region (S4's `panels_converge`).
    #[test]
    fn panels_converge() {
        let (kw, eps) = (C::new(10.0, 0.0), 1e-6);
        let n = lambda_rule(kw, eps, eps / 4.0, MAX_PANELS).panels;
        let a = panel_rule(kw, eps, n);
        let b = panel_rule(kw, eps, 2 * n);
        let sum = |r: &Rule, z: f64, rho: f64| -> C {
            r.t.iter()
                .zip(&r.lambda)
                .zip(&r.weight)
                .map(|((&t, &l), &w)| w * (-l * z).exp() * j_all(t * rho, 0)[0])
                .sum()
        };
        for z in [1.0, 2.5, 4.0] {
            for rho in [0.0, 2.0, 5.6] {
                let d = (sum(&a, z, rho) - sum(&b, z, rho)).norm();
                assert!(d <= eps / 4.0, "z {z} ρ {rho}: {d:.2e}");
            }
        }
    }

    /// The panel cap is reported: one panel per segment cannot meet 10⁻⁹ at κ = 20.
    #[test]
    fn a_capped_rule_says_so() {
        let fitted = lambda_rule(C::new(20.0, 0.0), 1e-9, 1e-9 / 4.0, 1);
        assert!(fitted.panels_capped);
    }
}

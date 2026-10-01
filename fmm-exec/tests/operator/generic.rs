//! The generic domain, a = (0.1, −2.3, 7.9) and w = 0.37, on level 16: the operator
//! against `nd-fmm-ref` at the absolute frames, with random (non-grid) points.
//!
//! Here the two sides no longer see the same geometry, and the difference is the
//! reference's, not the operator's:
//!
//! - **The reference frames carry ε |c| / r.** `geometry::centre` is within
//!   ε₆₄ (|a| + 2w) of the exact centre per component, and `direct` forms its shift
//!   (c′ − c)/r from two such centres, `leaf` its scaled points (x − c)/r from one. In
//!   units of the box that is up to β = ε₆₄ (|a| + 2w) / r_l per component, with
//!   |a| + 2w = 8.64 and r₁₆ = 0.37 / 2¹⁷: β ≈ 3.4e-10. The operator's tables and
//!   frames come from the integer keys and are exact (CONVENTIONS §3.12, §3.13).
//! - **The loading error is smaller.** The operator's leaf-scaled u is within
//!   ε₆₄ (2 |x − a| / r_l + |u|) of the exact one (§3.13, "Error bound"), at most
//!   2^(l − 51) + ε₆₄ ≈ 2.9e-11 at level 16, since |x − a| ≤ w, wherever the domain
//!   lies.
//!
//! A relative error δ in a shift of length |b| moves the harmonics of degree n by about
//! n δ, so a translation of degree p (shift harmonics up to 2p for M2L) differs by up to
//! about 2p β relative to its terms, and a leaf operator by about p β: the tolerance is
//! 2 (p + 1) β, 4.8e-9 at p = 6. The measured worst errors are 1–3 β.
//!
//! P2P has no frame: the reference subtracts absolute coordinates, which costs only
//! ε |x − y|, and the operator differs by its loading error. The separation u_t − ŷ in
//! units of the target leaf then moves by at most 2√3 (2^(l − 51) + ε₆₄), the potential
//! by at most that over the separation, relative, and the gradient by at most twice
//! that. The tolerance allows three times: 6√3 (2^(l − 51) + ε₆₄) / min |u_t − ŷ|,
//! relative to the terms.

use nd_fmm_exec::geometry::radius;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_ref::{Frame, Workspace, direct, leaf, p2p};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::morton;

use crate::common::{
    Basis, Kind, SplitMix64, Worst, absolute_frame, degree_error, dense, evaluation_terms,
    generic_domain, len, neighbours, norm, operator, p2p_terms, point_terms, random_coefficients,
    random_pair, random_points, source_chunk, split_output, sub, target_chunk, terms,
};

const LEVEL: usize = 16;
const P: usize = 6;

/// β = ε₆₄ (|a| + 2w) / r_l of the module documentation.
fn beta() -> f64 {
    let domain = generic_domain();
    let a = domain.lower();
    let largest = a.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
    f64::EPSILON / 2.0 * (largest + 2.0 * domain.side()) / radius(LEVEL, &domain)
}

#[test]
fn translations_in_a_generic_domain_differ_by_the_reference_frames() {
    // Error measure: per degree relative to the terms of the dense table (`common`).
    let domain = generic_domain();
    let mut rng = SplitMix64::new(0x7861);
    let tolerance = 2.0 * (P + 1) as f64 * beta();
    let tables = dense(P);
    let mut ws = Workspace::new(P);
    let [m2m, l2l, m2l] = ["M2M", "L2L", "M2L"].map(|op| {
        Worst::new(format!(
            "{op} vs direct, generic domain, level 16, p = {P} (terms; tolerance {tolerance:.1e})"
        ))
    });
    let mut op = operator(M2lStrategy::Dense, P, 0);
    for o in 0..8 {
        for _ in 0..4 {
            let parent = rng.key(LEVEL - 1);
            let child = morton::children(parent).unwrap()[o];
            let (from, to) = (
                absolute_frame(child, &domain),
                absolute_frame(parent, &domain),
            );
            let x = random_coefficients(Kind::Multipole, P, &mut rng);
            let (mut got, mut want) = (vec![0.0; len(P)], vec![0.0; len(P)]);
            op.m2m_pair(child, parent, &x, &mut got);
            direct::m2m(P, &from, &to, &mut ws, &x, &mut want);
            let e = degree_error(Kind::Multipole, P, &got, &want, &terms(&tables.m2m, o, &x));
            m2m.check(e, tolerance, || format!("octant {o}"));

            let x = random_coefficients(Kind::Local, P, &mut rng);
            let (mut got, mut want) = (vec![0.0; len(P)], vec![0.0; len(P)]);
            op.l2l_pair(parent, child, &x, &mut got);
            direct::l2l(P, &to, &from, &mut ws, &x, &mut want);
            let e = degree_error(Kind::Local, P, &got, &want, &terms(&tables.l2l, o, &x));
            l2l.check(e, tolerance, || format!("octant {o}"));
        }
    }
    for (index, d) in m2l_offsets().into_iter().enumerate() {
        let (source, target) = random_pair(LEVEL, d, &mut rng);
        let (from, to) = (
            absolute_frame(source, &domain),
            absolute_frame(target, &domain),
        );
        let x = random_coefficients(Kind::Multipole, P, &mut rng);
        let (mut got, mut want) = (vec![0.0; len(P)], vec![0.0; len(P)]);
        op.m2l_pair(source, target, &x, &mut got);
        direct::m2l(P, &from, &to, &mut ws, &x, &mut want);
        let e = degree_error(Kind::Local, P, &got, &want, &terms(&tables.m2l, index, &x));
        m2l.check(e, tolerance, || format!("d = {d:?}"));
    }
    // The reference's shifts are not exact here, unlike in the dyadic domain.
    assert!(
        m2l.get() > 1e3 * 1e-16,
        "the generic domain should show the frame error"
    );
}

#[test]
fn leaf_operators_in_a_generic_domain_differ_by_the_reference_frames() {
    // Error measures: P2M per degree relative to its terms; L2P potential and gradient
    // relative to their terms (`common`).
    let domain = generic_domain();
    let mut rng = SplitMix64::new(0x7862);
    let tolerance = 2.0 * (P + 1) as f64 * beta();
    let p2m = Worst::new(format!(
        "P2M vs leaf::p2m, generic domain, level 16, p = {P} (terms; tolerance {tolerance:.1e})"
    ));
    let l2p = Worst::new(format!(
        "L2P vs leaf::l2p, generic domain, level 16, p = {P}, potential and gradient (terms)"
    ));
    let mut op = operator(M2lStrategy::Rotation, P, 0);
    let mut ws = Workspace::new(P);
    for _ in 0..16 {
        let key = rng.key(LEVEL);
        let frame = absolute_frame(key, &domain);
        let (x, u) = random_points(key, &domain, 6, &mut rng);
        let q = rng.charges(6);
        let mut got = vec![0.0; len(P)];
        op.p2m_leaf(&source_chunk(&u, &q), &mut got);
        let mut want = vec![0.0; len(P)];
        leaf::p2m(P, &frame, &x, &q, &mut ws, &mut want);
        let scale = point_terms(Basis::Regular, P, &frame, &x, &q);
        let e = degree_error(Kind::Multipole, P, &got, &want, &scale);
        p2m.check(e, tolerance, || format!("key {key}"));

        let local = random_coefficients(Kind::Local, P, &mut rng);
        let mut output = vec![0.0; 4 * u.len()];
        op.l2p_leaf(&local, &target_chunk(&u), &mut output);
        let mut phi = vec![0.0; x.len()];
        let mut grad = vec![[0.0; 3]; x.len()];
        leaf::l2p(P, &frame, &local, &x, &mut ws, &mut phi, Some(&mut grad));
        let (potential, gradient) = split_output(&output);
        let unit = Frame::new([0.0; 3], 1.0);
        let r = frame.radius;
        for j in 0..x.len() {
            let (tp, tg) = evaluation_terms(Basis::Regular, P, &unit, &local, u[j]);
            l2p.check((potential[j] - r * phi[j]).abs() / tp, tolerance, || {
                format!("key {key}")
            });
            let want = grad[j].map(|g| r * r * g);
            l2p.check(norm(sub(gradient[j], want)) / tg, tolerance, || {
                format!("key {key}")
            });
        }
    }
}

#[test]
fn p2p_in_a_generic_domain_differs_by_the_loading_error() {
    // Error measure: potential and gradient relative to their terms (`common`).
    let domain = generic_domain();
    let mut rng = SplitMix64::new(0x7863);
    let loading = (2.0f64).powi(LEVEL as i32 - 51) + f64::EPSILON / 2.0;
    let worst = Worst::new("P2P vs p2p::p2p, generic domain, level 16 (terms)");
    let attained = Worst::new("P2P, generic domain, level 16, error / tolerance");
    let mut op = operator(M2lStrategy::Rotation, 0, 6);
    for _ in 0..4 {
        let target = rng.interior_key(LEVEL);
        let r_t = radius(LEVEL, &domain);
        let (x, u_t) = random_points(target, &domain, 6, &mut rng);
        for source in neighbours(target).into_iter().chain([target]) {
            let (y, u_s) = random_points(source, &domain, 6, &mut rng);
            let q = rng.charges(6);
            let mut output = vec![0.0; 4 * x.len()];
            op.p2p_pair(
                source,
                target,
                &source_chunk(&u_s, &q),
                &target_chunk(&u_t),
                &mut output,
            );
            let mut phi = vec![0.0; x.len()];
            let mut grad = vec![[0.0; 3]; x.len()];
            p2p::p2p(&y, &q, &x, &mut phi, Some(&mut grad));
            let (potential, gradient) = split_output(&output);
            for j in 0..x.len() {
                // The separation in units of the target leaf, from absolute coordinates.
                let closest = y
                    .iter()
                    .map(|&y| norm(sub(x[j], y)) / r_t)
                    .fold(f64::INFINITY, f64::min);
                let tolerance = 6.0 * 3f64.sqrt() * loading / closest;
                let (tp, tg) = p2p_terms(&y, &q, x[j]);
                let e = (potential[j] - r_t * phi[j]).abs() / (r_t * tp);
                worst.update(e);
                attained.update(e / tolerance);
                assert!(e <= tolerance, "{e:.3e} > {tolerance:.3e}");
                let want = grad[j].map(|g| r_t * r_t * g);
                let e = norm(sub(gradient[j], want)) / (r_t * r_t * tg);
                worst.update(e);
                attained.update(e / tolerance);
                assert!(e <= tolerance, "{e:.3e} > {tolerance:.3e}");
            }
        }
    }
}

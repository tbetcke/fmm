//! Level independence (C2.1): a table built at the canonical frames reproduces
//! `direct::m2m` and `direct::l2l` at the actual child and parent frames on every
//! level, exactly in a dyadic domain and within a documented rounding tolerance in a
//! generic one.

use std::time::Instant;

use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::{OCTANT_COUNT, octant_direction};
use nd_fmm_tables::{L2lTables, M2mTables, MatrixSet};

use crate::common::{
    DEBUG_DEGREES, DYADIC, Domain, GENERIC, Kind, LEVEL_TOL, LEVELS, P_MAX, SplitMix64, Worst,
    apply, child_index, degree_error, l2l, len, m2m, random_coefficients, terms,
};

/// Parent boxes on `level`: the first and the last box, and three random ones.
fn parent_boxes(level: u32, rng: &mut SplitMix64) -> Vec<[u64; 3]> {
    let last = (1u64 << level) - 1;
    let mut boxes = vec![[0; 3], [last; 3]];
    boxes.extend((0..3).map(|_| Domain::random_index(level, rng)));
    boxes
}

/// The table application to random coefficients against `direct` at the actual frames:
/// M2M from child `o` of box `index` on `level` to that box, and L2L the other way, at
/// the degree p of the tables.
/// Returns the per-degree errors relative to the terms of the table application.
pub fn check_box(
    domain: &Domain,
    level: u32,
    index: [u64; 3],
    o: usize,
    tables: (&MatrixSet<f64>, &MatrixSet<f64>),
    ws: &mut Workspace<f64>,
    rng: &mut SplitMix64,
) -> (f64, f64) {
    let p = tables.0.n().isqrt() - 1;
    let parent = domain.frame(level, index);
    let child = domain.frame(level + 1, child_index(index, o));

    let input = random_coefficients(Kind::Multipole, p, rng);
    let got = apply(tables.0, o, &input);
    let mut want = vec![0.0; len(p)];
    direct::m2m(p, &child, &parent, ws, &input, &mut want);
    let scale = terms(tables.0, o, &input);
    let m2m_error = degree_error(Kind::Multipole, p, &got, &want, &scale);

    let input = random_coefficients(Kind::Local, p, rng);
    let got = apply(tables.1, o, &input);
    let mut want = vec![0.0; len(p)];
    direct::l2l(p, &parent, &child, ws, &input, &mut want);
    let scale = terms(tables.1, o, &input);
    let l2l_error = degree_error(Kind::Local, p, &got, &want, &scale);
    (m2m_error, l2l_error)
}

/// Runs [`check_box`] in the dyadic domain on every parent level of `LEVELS`, for the
/// given degrees and table builders, and asserts `LEVEL_TOL`.
fn dyadic_levels(
    degrees: &[usize],
    tables: impl Fn(usize) -> (MatrixSet<f64>, MatrixSet<f64>),
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed);
    let mut ws = Workspace::new(*degrees.iter().max().unwrap());
    let worst_m2m = Worst::new("M2M table vs direct, dyadic domain, per degree (terms)");
    let worst_l2l = Worst::new("L2L table vs direct, dyadic domain, per degree (terms)");
    for &p in degrees {
        let (a, b) = tables(p);
        for level in LEVELS {
            for index in parent_boxes(level, &mut rng) {
                for o in 0..OCTANT_COUNT {
                    let (e_m2m, e_l2l) =
                        check_box(&DYADIC, level, index, o, (&a, &b), &mut ws, &mut rng);
                    worst_m2m.update(e_m2m);
                    worst_l2l.update(e_l2l);
                    assert!(
                        e_m2m <= LEVEL_TOL && e_l2l <= LEVEL_TOL,
                        "level {level}, box {index:?}, o = {o}, p = {p}: M2M {e_m2m:.3e}, L2L {e_l2l:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn tables_equal_direct_on_every_level_of_a_dyadic_domain() {
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for M2M,
    // Nₘ/Sₘ for L2L), relative to the terms |Aᵢₖ xₖ| of the table application
    // (`common`). The dyadic domain a = (−1.25, 0.5, 2), w = 3 makes every centre and
    // every centre difference exact, so `direct` sees b = t = ½ s_o and ρ = σ = ½
    // exactly, and only the summation order differs.
    dyadic_levels(
        &DEBUG_DEGREES,
        |p| (m2m(p).matrices().clone(), l2l(p).matrices().clone()),
        0x0c7a_0001,
    );
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn tables_equal_direct_on_every_level_at_large_p() {
    // As `tables_equal_direct_on_every_level_of_a_dyadic_domain`, for p = 20 and
    // p = 30 and all octants; also reports the build time of both families at p = 8,
    // 20 and 30.
    for p in [8, 20, P_MAX] {
        let start = Instant::now();
        let a = M2mTables::<f64>::build(p);
        let m2m_time = start.elapsed();
        let start = Instant::now();
        let b = L2lTables::<f64>::build(p);
        let l2l_time = start.elapsed();
        eprintln!("build p = {p}: M2M {m2m_time:.3?}, L2L {l2l_time:.3?}");
        if p > 8 {
            dyadic_levels(
                &[p],
                |_| (a.matrices().clone(), b.matrices().clone()),
                0x0c7a_0002 + p as u64,
            );
        }
    }
}

/// Worst-case rounding of a scaled child–parent shift in `domain` on parent `level`,
/// per component (see `generic_domain_error_is_bounded_by_the_centre_rounding`).
fn shift_rounding_bound(domain: &Domain, level: u32) -> f64 {
    let reach = domain
        .corner
        .iter()
        .map(|a| a.abs() + domain.side)
        .fold(0.0, f64::max);
    let r = domain.side / (1u64 << (level + 1)) as f64;
    2.0 * f64::EPSILON * reach / r + f64::EPSILON
}

/// The largest component of the scaled shift of `direct` minus the exact ½ s_o.
fn shift_error(from_scaled: [f64; 3], o: usize) -> f64 {
    let s = octant_direction(o);
    (0..3)
        .map(|a| (from_scaled[a] - 0.5 * s[a] as f64).abs())
        .fold(0.0, f64::max)
}

#[test]
fn generic_domain_error_is_bounded_by_the_centre_rounding() {
    // The generic domain a = (0.1, −2.3, 7.9), w = 0.37, parent level 15, against the
    // dyadic one on the same level.
    //
    // Tolerance. Each centre c = a + (i + ½) w/2^l is formed with two roundings, so it
    // is off by at most ε (|a| + w) per component. The child–parent difference is then
    // exact (Sterbenz: the two centres agree to within a factor 2), and dividing by
    // r_l adds ε/2 · ½. So `direct` sees b = ½ s_o + δ (M2M) and t = ½ s_o + δ (L2L)
    // with, per component,
    //
    //   |δ| ≤ δ_max = 2 ε (|a| + w)_max / r_l + ε,
    //
    // about 6.5e-10 here: an error of ε |c| / r_l, which grows as 2^l. The table has the
    // exact shift. By the addition theorem Rₙ(b + δ) − Rₙ(b) = Σ_{q≥1} R_q(δ) Rₙ₋q(b),
    // whose weighted norm is at most (|b| + |δ|)ⁿ − |b|ⁿ ≈ n |δ| |b|ⁿ⁻¹, so each term of
    // degree n of the output moves by at most n |δ|/|b| of itself, with n ≤ p. With
    // |b| = √3/2 and |δ| ≤ √3 δ, and a factor 4 for the Cauchy–Schwarz sum over orders
    // and the rounding of the direct sum, the per-degree error relative to the terms is
    // at most
    //
    //   tol = LEVEL_TOL + 4 · 2 p δ,
    //
    // with δ the actual largest component of the shift error of each pair, which the
    // test also checks against δ_max.
    //
    // Error measure: coefficients per degree in the orthonormal weighting, relative to
    // the terms |Aᵢₖ xₖ| of the table application (`common`).
    let level = 15;
    let mut rng = SplitMix64::new(0x0c7a_0003);
    let mut ws = Workspace::new(DEBUG_DEGREES[DEBUG_DEGREES.len() - 1]);
    let delta_max = shift_rounding_bound(&GENERIC, level);
    let worst_delta = Worst::new("generic domain, level 15: largest shift error δ");
    let generic = [
        Worst::new("M2M table vs direct, generic domain, level 15, per degree (terms)"),
        Worst::new("L2L table vs direct, generic domain, level 15, per degree (terms)"),
    ];
    let dyadic = [
        Worst::new("M2M table vs direct, dyadic domain, level 15, per degree (terms)"),
        Worst::new("L2L table vs direct, dyadic domain, level 15, per degree (terms)"),
    ];
    let worst_ratio = Worst::new("generic domain, level 15: error / tolerance");
    eprintln!("generic domain, level 15: a-priori δ_max = {delta_max:.3e}");
    for p in DEBUG_DEGREES {
        let tables = (m2m(p).matrices(), l2l(p).matrices());
        for index in parent_boxes(level, &mut rng) {
            for o in 0..OCTANT_COUNT {
                let parent = GENERIC.frame(level, index);
                let child = GENERIC.frame(level + 1, child_index(index, o));
                // b = (c − c′)/r′ for M2M and t = (c′ − c)/r for L2L, as `direct`
                // forms them: both are (c_child − c_parent)/r_parent.
                let delta = shift_error(parent.scaled(child.centre), o);
                worst_delta.update(delta);
                assert!(
                    delta <= delta_max,
                    "δ = {delta:.3e} > δ_max = {delta_max:.3e}"
                );
                let tol = LEVEL_TOL + 8.0 * p as f64 * delta;
                let (e_m2m, e_l2l) =
                    check_box(&GENERIC, level, index, o, tables, &mut ws, &mut rng);
                generic[0].update(e_m2m);
                generic[1].update(e_l2l);
                worst_ratio.update(e_m2m.max(e_l2l) / tol);
                assert!(
                    e_m2m <= tol && e_l2l <= tol,
                    "box {index:?}, o = {o}, p = {p}: M2M {e_m2m:.3e}, L2L {e_l2l:.3e}, tol {tol:.3e}"
                );
                let (d_m2m, d_l2l) = check_box(&DYADIC, level, index, o, tables, &mut ws, &mut rng);
                dyadic[0].update(d_m2m);
                dyadic[1].update(d_l2l);
                assert!(d_m2m <= LEVEL_TOL && d_l2l <= LEVEL_TOL);
            }
        }
    }
    assert!(
        worst_delta.get() > 0.0,
        "the generic domain should round its centres"
    );
}

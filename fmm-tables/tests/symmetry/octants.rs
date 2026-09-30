//! The octant identities of CONVENTIONS §3.12, "Operator identities":
//! M2M(P s) = T_M(P) M2M(s) T_M(Pᵀ) and L2L(P s) = T_L(P) L2L(s) T_L(Pᵀ), from octant 0
//! to every octant, against the dense T3 tables. A test only: the octant tables stay
//! dense.

use nd_fmm_tables::geometry::{OCTANT_COUNT, octant_direction};
use nd_fmm_tables::symmetry::{CoefficientTransform, SignedPermutation};
use nd_fmm_tables::{L2lTables, M2mTables};

use crate::common::{CLASS_TOL, Kind, Worst, block_error, product};

/// The child index o with s_o = `s`.
fn octant_of(s: [i64; 3]) -> usize {
    (0..OCTANT_COUNT)
        .find(|&o| octant_direction(o) == s)
        .expect("a sign vector")
}

/// Checks both identities for all 48 P at degree `p`; returns the worst M2M and L2L
/// errors.
pub fn check_octants(p: usize, m2m: &M2mTables<f64>, l2l: &L2lTables<f64>) -> (f64, f64) {
    let s0 = octant_direction(0);
    let (mut worst_m, mut worst_l): (f64, f64) = (0.0, 0.0);
    for e in SignedPermutation::all() {
        let o = octant_of(e.apply(s0));
        let n2 = m2m.matrices().n().pow(2);
        let (tm, tm_inv) = (
            CoefficientTransform::multipole(p, e),
            CoefficientTransform::multipole(p, e.transpose()),
        );
        let (got, terms) = product(&tm, m2m.matrices(), 0, &tm_inv);
        let want = m2m.matrices().matrix(o);
        assert_eq!(want.len(), n2);
        let err = block_error(Kind::Multipole, Kind::Multipole, p, &got, want, &terms);
        assert!(
            err <= CLASS_TOL,
            "M2M, p = {p}, g = {}, o = {o}: {err:.3e}",
            e.index()
        );
        worst_m = worst_m.max(err);

        let (tl, tl_inv) = (
            CoefficientTransform::local(p, e),
            CoefficientTransform::local(p, e.transpose()),
        );
        let (got, terms) = product(&tl, l2l.matrices(), 0, &tl_inv);
        let err = block_error(
            Kind::Local,
            Kind::Local,
            p,
            &got,
            l2l.matrices().matrix(o),
            &terms,
        );
        assert!(
            err <= CLASS_TOL,
            "L2L, p = {p}, g = {}, o = {o}: {err:.3e}",
            e.index()
        );
        worst_l = worst_l.max(err);
    }
    (worst_m, worst_l)
}

#[test]
fn transforms_map_the_octant_0_tables_to_every_octant() {
    // All 48 P, each mapping octant 0 to the octant of P s₀, p ≤ 12.
    //
    // Error measure: per block of output and input degree, in the orthonormal
    // weighting of both sides (Nₘ for M2M, Nₘ/Sₘ for L2L), relative to the terms
    // |T(P)| |A₀| |T(Pᵀ)| of the three products (`block_error`).
    let worst_m = Worst::new("M2M(P s₀) vs T_M(P) M2M(s₀) T_M(Pᵀ), p ≤ 12, per block (terms)");
    let worst_l = Worst::new("L2L(P s₀) vs T_L(P) L2L(s₀) T_L(Pᵀ), p ≤ 12, per block (terms)");
    for p in [0, 1, 2, 3, 5, 8, 12] {
        let (m, l) = check_octants(p, &M2mTables::build(p), &L2lTables::build(p));
        worst_m.update(m);
        worst_l.update(l);
    }
}

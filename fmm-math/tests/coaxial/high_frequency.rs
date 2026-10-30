//! Tests 3 (projections), 7 and 8: whole tables at high frequency, in release mode. The
//! fixture's samples cannot show the recursion's instability (S2 §4.3); these compare
//! independent methods over every entry, in the FMM weights.

use nd_fmm_math::Complex;
use nd_fmm_math::coaxial::CoaxialTable;

use crate::support::{
    DISTANCES2, KINDS, degree, m2l_t, m2l_weights, scale, shift_weights, table_difference, weighted,
};

/// The bound on the weighted error (S2's question 2, signed off 2026-10-27).
const WEIGHTED: f64 = 1e-11;

/// The outer projection's nodes, as [`CoaxialTable::m2m`].
fn outer_nodes(p_child: usize, p_parent: usize) -> usize {
    p_child.max(p_parent) + 40
}

/// The inner projection's nodes: max(p) + 40 + ⌈|k| t⌉, as S7 for its L2L tables.
fn inner_nodes(k: Complex<f64>, t: f64, p_child: usize, p_parent: usize) -> usize {
    p_child.max(p_parent) + 40 + (k.norm() * t).ceil() as usize
}

/// Test 3, the projections: for real k at k a = 100 and 300 the outer projection's (S|S)
/// and the inner projection's (R|R) have imaginary parts below 1e-11 in the shift
/// weights (their arithmetic is complex throughout, so this can fail), read as M2M and
/// as L2L.
#[test]
#[ignore = "release mode: tables to p = 371 by projection"]
fn real_k_projected_shift_coefficients_are_real() {
    let t = 0.5;
    let mut worst = (0.0f64, String::new());
    for ka in [100.0, 300.0] {
        let (pp, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
        let k = Complex::new(ka, 0.0);
        let (child, parent) = shift_weights(k, t, pc, pp);
        let outer = CoaxialTable::projection_regular_outer(
            k,
            t,
            8.0 * t / 3.0f64.sqrt(),
            pc,
            pp,
            outer_nodes(pc, pp),
        );
        let inner =
            CoaxialTable::projection_regular_inner(k, t, t, pc, pp, inner_nodes(k, t, pc, pp));
        for (table, what) in [(outer, "outer"), (inner, "inner")] {
            let real = |m, a, b| Complex::new(table.get(m, a, b).re, 0.0);
            let m2m = weighted(pc, pp, &child, &parent, |m, a, b| table.get(m, a, b), real);
            let l2l = table.clone().reversed();
            let real = |m, a, b| Complex::new(l2l.get(m, a, b).re, 0.0);
            let l2l = weighted(pp, pc, &parent, &child, |m, a, b| l2l.get(m, a, b), real);
            for (err, read) in [(m2m, "M2M"), (l2l, "L2L")] {
                println!("ka={ka} {what} projection read as {read}: imaginary parts {err:.2e}");
                if err > worst.0 {
                    worst = (err, format!("ka={ka} {what} {read}"));
                }
            }
        }
    }
    assert!(worst.0 <= WEIGHTED, "{worst:?}");
}

/// Test 7 (S2's test 6): at the threshold Re(k) a = 40 (p = 77), every kind of k, the
/// recursion against the projection over whole tables in the FMM weights: M2M (outer),
/// L2L (inner) and the M2L at all 15 V-list distances.
#[test]
#[ignore = "release mode: 66 tables to p = 77 by projection"]
fn recursion_is_stable_at_the_threshold() {
    let ka = 40.0;
    let (pp, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
    let t = 0.5;
    let mut worst = (0.0f64, String::new());
    let mut note = |err: f64, what: String| {
        if err > worst.0 {
            worst = (err, what);
        }
    };
    for (name, re, im) in KINDS {
        let k = Complex::new(re * ka, im * ka);
        let (child, parent) = shift_weights(k, t, pc, pp);
        let rec = CoaxialTable::regular(k * t, pc, pp);
        let outer = CoaxialTable::projection_regular_outer(
            k,
            t,
            8.0 * t / 3.0f64.sqrt(),
            pc,
            pp,
            outer_nodes(pc, pp),
        );
        note(
            table_difference(&outer, &rec, &child, &parent),
            format!("{name} M2M"),
        );
        let inner =
            CoaxialTable::projection_regular_inner(k, t, t, pp, pc, inner_nodes(k, t, pc, pp));
        note(
            table_difference(&inner, &rec.reversed(), &parent, &child),
            format!("{name} L2L"),
        );
        let s = scale(k.norm());
        let w = m2l_weights(k, 1.0, s, pp);
        for d2 in DISTANCES2 {
            let t = m2l_t(d2);
            let rec = CoaxialTable::singular_regular(k * t, s, pp, pp);
            let proj = CoaxialTable::projection_singular_regular(k, t, 1.0, s, pp, pp, pp + 40);
            note(
                table_difference(&proj, &rec, &w, &w),
                format!("{name} M2L d²={d2}"),
            );
        }
    }
    println!(
        "k a = 40, p {pc} -> {pp}: worst weighted {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= WEIGHTED, "{worst:?}");
}

/// Test 8: the L2L table above the threshold. At each k a of the parent and kind of k,
/// the recursion, the outer projection (computed in M2M's orientation, read reversed) and
/// the inner projection (computed in L2L's), against each other over whole tables in the
/// shift weights read as L2L, and the inner projection against itself with 60 more nodes.
/// The two projections are independent and must agree within the bound everywhere; the
/// recursion must agree with the inner projection where S2 found it stable (k a ≤ 40,
/// and Im k = Re k and Yukawa throughout).
#[test]
#[ignore = "release mode: tables to p = 371 by projection"]
fn l2l_tables_above_the_threshold() {
    let t = 0.5;
    println!(
        "{:<11} {:>5} {:>9} {:>11} {:>11} {:>11} {:>11}",
        "kind", "ka", "p_P->p_c", "rec-outer", "rec-inner", "outer-inner", "inner-nodes"
    );
    for ka in [40.0, 64.0, 100.0, 128.0, 200.0, 300.0] {
        let (pp, pc) = (degree(ka, 1e-10), degree(ka / 2.0, 1e-10));
        for (name, re, im) in KINDS {
            let k = Complex::new(re * ka, im * ka);
            let (child, parent) = shift_weights(k, t, pc, pp);
            let rec = CoaxialTable::regular(k * t, pp, pc);
            let outer = CoaxialTable::projection_regular_outer(
                k,
                t,
                8.0 * t / 3.0f64.sqrt(),
                pc,
                pp,
                outer_nodes(pc, pp),
            )
            .reversed();
            let nodes = inner_nodes(k, t, pc, pp);
            let inner = CoaxialTable::projection_regular_inner(k, t, t, pp, pc, nodes);
            let more = CoaxialTable::projection_regular_inner(k, t, t, pp, pc, nodes + 60);
            let d = |a: &CoaxialTable<f64>, b: &CoaxialTable<f64>| {
                table_difference(a, b, &parent, &child)
            };
            let (rec_outer, rec_inner) = (d(&rec, &outer), d(&rec, &inner));
            let (outer_inner, nodes_inner) = (d(&outer, &inner), d(&more, &inner));
            println!(
                "{name:<11} {ka:>5} {pp:>4}->{pc:<4} {rec_outer:>11.2e} {rec_inner:>11.2e} \
                 {outer_inner:>11.2e} {nodes_inner:>11.2e}"
            );
            let what = format!("{name} ka={ka}");
            assert!(
                outer_inner <= WEIGHTED,
                "{what}: outer-inner {outer_inner:.2e}"
            );
            assert!(nodes_inner <= WEIGHTED, "{what}: nodes {nodes_inner:.2e}");
            if ka <= 40.0 || name == "complex1" || name == "imaginary" {
                assert!(rec_inner <= WEIGHTED, "{what}: rec-inner {rec_inner:.2e}");
            }
        }
    }
}

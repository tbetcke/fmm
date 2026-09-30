//! Random properties in the dyadic domain: random levels, boxes, octants, degrees,
//! coefficients and sources, 64 cases each. Each property reuses the error measure and
//! tolerance of the deterministic test it generalises, and prints its worst error over
//! the cases with `--nocapture`.

use nd_fmm_ref::{Workspace, leaf};
use proptest::prelude::*;

use crate::common::{
    CHAIN_TOL, DYADIC, Kind, LEVEL_TOL, P_DEBUG, SplitMix64, Worst, apply, child_index,
    degree_error, l2l, len, m2m, place, terms,
};
use crate::levels::check_box;

/// 64 cases per property.
fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

/// A parent level 0 ≤ l ≤ 15 and a box index on it.
fn parent_box() -> impl Strategy<Value = (u32, [u64; 3])> {
    (0u32..=15).prop_flat_map(|level| {
        let n = 1u64 << level;
        (Just(level), [0..n, 0..n, 0..n])
    })
}

#[test]
fn tables_equal_direct_for_random_boxes() {
    let worst_m2m = Worst::new("property: M2M table vs direct, dyadic domain (terms)");
    let worst_l2l = Worst::new("property: L2L table vs direct, dyadic domain (terms)");
    proptest!(config(), |(
        (level, index) in parent_box(),
        o in 0usize..8,
        p in 0usize..=P_DEBUG,
        seed in any::<u64>(),
    )| {
        // Error measure: as `levels::tables_equal_direct_on_every_level_of_a_dyadic_domain`.
        let mut ws = Workspace::new(p);
        let mut rng = SplitMix64::new(seed);
        let tables = (m2m(p).matrices(), l2l(p).matrices());
        let (e_m2m, e_l2l) = check_box(&DYADIC, level, index, o, tables, &mut ws, &mut rng);
        worst_m2m.update(e_m2m);
        worst_l2l.update(e_l2l);
        prop_assert!(e_m2m <= LEVEL_TOL, "M2M: {e_m2m:.3e}");
        prop_assert!(e_l2l <= LEVEL_TOL, "L2L: {e_l2l:.3e}");
    });
}

#[test]
fn m2m_table_after_p2m_equals_p2m_at_the_parent_for_random_boxes() {
    let worst = Worst::new("property: P2M, M2M table vs P2M at parent (terms)");
    proptest!(config(), |(
        (level, index) in parent_box(),
        o in 0usize..8,
        p in 0usize..=P_DEBUG,
        sources in prop::collection::vec(
            ([-1.0..=1.0f64, -1.0..=1.0f64, -1.0..=1.0f64], prop_oneof![-1.0..-0.01f64, 0.01..1.0f64]),
            1..=6,
        ),
    )| {
        // Error measure: as `chains::m2m_table_after_p2m_equals_p2m_at_the_parent`.
        let mut ws = Workspace::new(p);
        let parent = DYADIC.frame(level, index);
        let child = DYADIC.frame(level + 1, child_index(index, o));
        let (y, q): (Vec<[f64; 3]>, Vec<f64>) =
            sources.iter().map(|&(u, q)| (place(&child, u), q)).unzip();
        let mut at_child = vec![0.0; len(p)];
        leaf::p2m(p, &child, &y, &q, &mut ws, &mut at_child);
        let got = apply(m2m(p).matrices(), o, &at_child);
        let mut want = vec![0.0; len(p)];
        leaf::p2m(p, &parent, &y, &q, &mut ws, &mut want);
        let scale = terms(m2m(p).matrices(), o, &at_child);
        let e = worst.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
        prop_assert!(e <= CHAIN_TOL, "{e:.3e}");
    });
}

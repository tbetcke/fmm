//! Random properties in the dyadic domain: random levels, V-list pairs, offsets,
//! degrees, multipoles and sources, 64 cases each. Each property reuses the error
//! measure and tolerance of the deterministic test it generalises, and prints its worst
//! error over the cases with `--nocapture`.

use nd_fmm_ref::{Workspace, direct, leaf};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use proptest::prelude::*;

use crate::common::{
    CHAIN_TOL, DEBUG_DEGREES, DYADIC, LEVEL_TOL, SplitMix64, Worst, apply, l2p_terms, len, m2l,
    place, terms,
};
use crate::levels::check_pair;

/// 64 cases per property.
fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

/// A level 2 ≤ l ≤ 16, a table index and a V-list pair (source, target) on that level
/// at its offset, with both boxes in the domain.
fn v_list_pair() -> impl Strategy<Value = (u32, usize, ([u64; 3], [u64; 3]))> {
    (2u32..=16, 0..M2L_OFFSET_COUNT).prop_flat_map(|(level, index)| {
        let d = m2l_offsets()[index];
        let n = 1i64 << level;
        // Along each axis, the targets whose source lies in the domain.
        let axis = |t: i64| (t.max(0) as u64)..=((n - 1).min(n - 1 + t) as u64);
        let target = [axis(d[0]), axis(d[1]), axis(d[2])];
        (Just(level), Just(index), target).prop_map(move |(level, index, target)| {
            let source = core::array::from_fn(|a| (target[a] as i64 - d[a]) as u64);
            (level, index, (source, target))
        })
    })
}

/// A degree of the debug run, whose table is cached.
fn debug_degree() -> impl Strategy<Value = usize> {
    prop::sample::select(DEBUG_DEGREES.to_vec())
}

#[test]
fn tables_equal_direct_for_random_pairs() {
    let worst = Worst::new("property: M2L table vs direct, dyadic domain (terms)");
    proptest!(config(), |(
        (level, index, pair) in v_list_pair(),
        p in debug_degree(),
        seed in any::<u64>(),
    )| {
        // Error measure: as `levels::tables_equal_direct_on_levels_2_9_16_of_a_dyadic_domain`.
        let mut ws = Workspace::new(p);
        let mut rng = SplitMix64::new(seed);
        let d = m2l_offsets()[index];
        let e = check_pair(level, pair, d, m2l(p).matrices(), index, &mut ws, &mut rng);
        worst.update(e);
        prop_assert!(e <= LEVEL_TOL, "{e:.3e}");
    });
}

#[test]
fn p2m_table_l2p_equals_the_chain_with_direct_m2l_for_random_pairs() {
    let worst = Worst::new("property: P2M, M2L table, L2P vs direct M2L chain (terms)");
    let unit = || [-1.0..=1.0f64, -1.0..=1.0f64, -1.0..=1.0f64];
    proptest!(config(), |(
        (level, index, (s, t)) in v_list_pair(),
        p in debug_degree(),
        sources in prop::collection::vec(
            (unit(), prop_oneof![-1.0..-0.01f64, 0.01..1.0f64]),
            1..=6,
        ),
        targets in prop::collection::vec(unit(), 1..=4),
    )| {
        // Error measure: as `chains::p2m_table_l2p_equals_the_chain_with_direct_m2l`.
        let mut ws = Workspace::new(p);
        let (source, target) = (DYADIC.frame(level, s), DYADIC.frame(level, t));
        let (y, q): (Vec<[f64; 3]>, Vec<f64>) =
            sources.iter().map(|&(u, q)| (place(&source, u), q)).unzip();
        let x: Vec<[f64; 3]> = targets.iter().map(|&v| place(&target, v)).collect();
        let mut multipole = vec![0.0; len(p)];
        leaf::p2m(p, &source, &y, &q, &mut ws, &mut multipole);
        let set = m2l(p).matrices();
        let from_table = apply(set, index, &multipole);
        let mut from_direct = vec![0.0; len(p)];
        direct::m2l(p, &source, &target, &mut ws, &multipole, &mut from_direct);
        let mut got = vec![0.0; x.len()];
        leaf::l2p(p, &target, &from_table, &x, &mut ws, &mut got, None);
        let mut want = vec![0.0; x.len()];
        leaf::l2p(p, &target, &from_direct, &x, &mut ws, &mut want, None);
        let tau = terms(set, index, &multipole);
        for ((g, w), &x) in got.iter().zip(&want).zip(&x) {
            let e = worst.update((g - w).abs() / l2p_terms(p, &target, &tau, x));
            prop_assert!(e <= CHAIN_TOL, "{e:.3e}");
        }
    });
}

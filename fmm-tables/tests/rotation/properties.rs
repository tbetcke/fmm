//! Random properties: random offsets, octants, levels, boxes, degrees and coefficients,
//! 64 cases each. Each property reuses the error measure and tolerance of the
//! deterministic test it generalises, and prints its worst error over the cases with
//! `--nocapture`.

use nd_fmm_ref::Workspace;
use nd_fmm_tables::rotation::Operator;
use proptest::prelude::*;

use crate::common::{
    DIRECT_TOL, P_DEBUG, REFERENCE_TOL, SplitMix64, Worst, apply, dense, error, input_kind,
    random_coefficients, reference, run, tables, translate,
};
use crate::levels::check_level;

/// 64 cases per property.
fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

/// A family and one of its entries: a table index for M2L, a child index otherwise.
fn entry() -> impl Strategy<Value = (Operator, usize)> {
    prop_oneof![
        (0usize..316).prop_map(|t| (Operator::M2l, t)),
        (0usize..8).prop_map(|t| (Operator::M2m, t)),
        (0usize..8).prop_map(|t| (Operator::L2l, t)),
    ]
}

#[test]
fn operators_equal_reference_and_direct_for_random_entries() {
    let worst_reference = Worst::new("property: tables vs rotation, canonical frames (terms)");
    let worst_direct = Worst::new("property: tables vs dense direct, canonical frames (terms)");
    proptest!(config(), |(
        (op, t) in entry(),
        p in 0usize..=P_DEBUG,
        seed in any::<u64>(),
    )| {
        // Error measure: as `tables::operators_equal_reference_rotation_at_the_canonical_frames`
        // and `direct::operators_equal_direct_through_the_dense_tables`.
        let mut rng = SplitMix64::new(seed);
        let mut ws = Workspace::new(p);
        let (from, to) = op.frames(t);
        let set = dense(p).of(op);
        let x = random_coefficients(input_kind(op), p, &mut rng);
        let got = run(tables(p).tables(op), t, &x);
        let want = translate(reference(op), p, (&from, &to), &mut ws, &x);
        let e = worst_reference.update(error(op, set, t, &x, &got, &want));
        prop_assert!(e <= REFERENCE_TOL, "reference: {e:.3e}");
        let e = worst_direct.update(error(op, set, t, &x, &got, &apply(set, t, &x)));
        prop_assert!(e <= DIRECT_TOL, "direct: {e:.3e}");
    });
}

#[test]
fn operators_equal_reference_for_random_levels_and_boxes() {
    let worst = Worst::new("property: tables vs rotation, random levels, dyadic domain (terms)");
    proptest!(config(), |(
        (op, t) in entry(),
        level in 2u32..=16,
        p in 0usize..=P_DEBUG,
        seed in any::<u64>(),
    )| {
        // Error measure: as `levels::operators_equal_reference_rotation_on_levels_2_9_16_of_a_dyadic_domain`.
        let mut rng = SplitMix64::new(seed);
        let mut ws = Workspace::new(p);
        let e = check_level(tables(p), dense(p).of(op), op, level, t, &mut ws, &mut rng);
        worst.update(e);
        prop_assert!(e <= REFERENCE_TOL, "{e:.3e}");
    });
}

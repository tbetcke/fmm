//! Large degrees, in ignored tests for the release run: agreement with
//! `nd_fmm_ref::rotation` and `direct` at p = 12 and 20, levels 2, 9 and 16 at those
//! degrees, M2M and L2L for 20 < p ≤ 30, and the build times and storage at p = 8, 16
//! and 20.

use std::time::Instant;

use nd_fmm_ref::Workspace;
use nd_fmm_tables::RotationTables;
use nd_fmm_tables::rotation::Operator;

use crate::common::{
    DIRECT_TOL, Dense, HIGH_DEGREE_TOL, OPERATORS, P_MAX, P_MAX_M2L, REFERENCE_TOL, SplitMix64,
    Worst, apply, error, input_kind, random_coefficients, reference, run, translate,
};
use crate::levels::check_levels;

/// Names of the worst-error trackers at degree `p`, per family and measure.
fn names(p: usize) -> [&'static str; 9] {
    match p {
        12 => [
            "M2L tables vs rotation, canonical frames, p = 12 (terms)",
            "M2M tables vs rotation, canonical frames, p = 12 (terms)",
            "L2L tables vs rotation, canonical frames, p = 12 (terms)",
            "M2L tables vs dense direct, p = 12 (terms)",
            "M2M tables vs dense direct, p = 12 (terms)",
            "L2L tables vs dense direct, p = 12 (terms)",
            "M2L tables vs rotation, levels 2, 9, 16, p = 12 (terms)",
            "M2M tables vs rotation, levels 2, 9, 16, p = 12 (terms)",
            "L2L tables vs rotation, levels 2, 9, 16, p = 12 (terms)",
        ],
        _ => [
            "M2L tables vs rotation, canonical frames, p = 20 (terms)",
            "M2M tables vs rotation, canonical frames, p = 20 (terms)",
            "L2L tables vs rotation, canonical frames, p = 20 (terms)",
            "M2L tables vs dense direct, p = 20 (terms)",
            "M2M tables vs dense direct, p = 20 (terms)",
            "L2L tables vs dense direct, p = 20 (terms)",
            "M2L tables vs rotation, levels 2, 9, 16, p = 20 (terms)",
            "M2M tables vs rotation, levels 2, 9, 16, p = 20 (terms)",
            "L2L tables vs rotation, levels 2, 9, 16, p = 20 (terms)",
        ],
    }
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn operators_equal_reference_and_direct_at_p_12_and_20() {
    // Error measures: as `tables::operators_equal_reference_rotation_at_the_canonical_frames`
    // (1e-14), `direct::operators_equal_direct_through_the_dense_tables` (1e-13, through
    // the dense tables of T3 and T4 only) and
    // `levels::operators_equal_reference_rotation_on_levels_2_9_16_of_a_dyadic_domain`
    // (1e-14, one pair per entry and level). All 316 offsets and all 8 octants, two
    // random inputs each at the canonical frames.
    for p in [12, P_MAX_M2L] {
        let rotation = RotationTables::<f64>::build(p);
        // The dense tables live for the rest of the test binary.
        let dense: &'static Dense = Box::leak(Box::new(Dense::build(p)));
        let [r0, r1, r2, d0, d1, d2, l0, l1, l2] = names(p).map(Worst::new);
        let mut rng = SplitMix64::new(0x0706_0401 + p as u64);
        let mut ws = Workspace::new(p);
        for (op, (worst_reference, worst_direct)) in
            OPERATORS
                .into_iter()
                .zip([(&r0, &d0), (&r1, &d1), (&r2, &d2)])
        {
            let set = dense.of(op);
            for t in 0..op.count() {
                let (from, to) = op.frames(t);
                for _ in 0..2 {
                    let x = random_coefficients(input_kind(op), p, &mut rng);
                    let got = run(rotation.tables(op), t, &x);
                    let want = translate(reference(op), p, (&from, &to), &mut ws, &x);
                    let e = worst_reference.update(error(op, set, t, &x, &got, &want));
                    assert!(e <= REFERENCE_TOL, "{op:?}, p = {p}, entry {t}: {e:.3e}");
                    let e = worst_direct.update(error(op, set, t, &x, &got, &apply(set, t, &x)));
                    assert!(e <= DIRECT_TOL, "{op:?}, p = {p}, entry {t}: {e:.3e}");
                }
            }
        }
        check_levels(&rotation, |op| dense.of(op), 1, &[l0, l1, l2], &mut rng);
    }
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn octant_operators_above_p_20() {
    // Error measure: output coefficients per degree in the orthonormal weighting,
    // relative to the terms of the dense M2M or L2L table (`main`), bound 1e-11 for
    // 20 < p ≤ 30 (brief T6, as in Phase 1 T6), against `nd_fmm_ref::rotation` and
    // against the dense tables of T3 (`direct`). All 8 octants, four random inputs each.
    let worst = [
        Worst::new("M2M tables vs rotation, p = 25, 30 (terms)"),
        Worst::new("L2L tables vs rotation, p = 25, 30 (terms)"),
        Worst::new("M2M tables vs dense direct, p = 25, 30 (terms)"),
        Worst::new("L2L tables vs dense direct, p = 25, 30 (terms)"),
    ];
    let mut rng = SplitMix64::new(0x0706_0402);
    for p in [25, P_MAX] {
        let rotation = RotationTables::<f64>::build(p);
        let dense = [
            nd_fmm_tables::M2mTables::<f64>::build(p).matrices().clone(),
            nd_fmm_tables::L2lTables::<f64>::build(p).matrices().clone(),
        ];
        let mut ws = Workspace::new(p);
        for (k, op) in [Operator::M2m, Operator::L2l].into_iter().enumerate() {
            let set = &dense[k];
            for o in 0..op.count() {
                let (from, to) = op.frames(o);
                for _ in 0..4 {
                    let x = random_coefficients(input_kind(op), p, &mut rng);
                    let got = run(rotation.tables(op), o, &x);
                    let want = translate(reference(op), p, (&from, &to), &mut ws, &x);
                    let e = worst[k].update(error(op, set, o, &x, &got, &want));
                    assert!(e <= HIGH_DEGREE_TOL, "{op:?}, p = {p}, o = {o}: {e:.3e}");
                    let e = worst[2 + k].update(error(op, set, o, &x, &got, &apply(set, o, &x)));
                    assert!(e <= HIGH_DEGREE_TOL, "{op:?}, p = {p}, o = {o}: {e:.3e}");
                }
            }
        }
    }
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn build_time_and_storage_report() {
    // Reports, without asserting, the build time and the storage of the tables at
    // p = 8, 16 and 20, per family and in all, in f64. T8 measures them properly.
    for p in [8, 16, P_MAX_M2L] {
        let start = Instant::now();
        let rotation = RotationTables::<f64>::build(p);
        let elapsed = start.elapsed();
        let mb = |reals: usize| 8.0 * reals as f64 / 1e6;
        eprintln!(
            "rotation tables p = {p}: build {elapsed:.3?}, storage {:.3} MB \
             (M2L {:.3} MB, M2M {:.3} MB, L2L {:.3} MB)",
            mb(rotation.storage_len()),
            mb(rotation.tables(Operator::M2l).storage_len()),
            mb(rotation.tables(Operator::M2m).storage_len()),
            mb(rotation.tables(Operator::L2l).storage_len()),
        );
    }
}

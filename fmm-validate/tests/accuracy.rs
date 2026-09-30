//! Smoke runs of the accuracy sweep at p ≤ 4 with few points, on the reference and the
//! table path.
//!
//! These check that the sweep runs and is consistent, not the accuracy figures
//! themselves: those are reported by the `accuracy` example. Error measure throughout:
//! the relative L2 and max errors against `direct_sum` that the sweep reports
//! (`nd_fmm_validate::metrics::ErrorNorms`), compared as plain numbers.

use nd_fmm_validate::accuracy::{Chain, Config, Errors, Row, Translations, sweep};

const SMOKE: Config = Config {
    sources: 40,
    targets: 30,
    seed: 7,
};

const P_MAX: usize = 4;

fn values(e: &Errors) -> [f64; 4] {
    [
        e.potential.l2,
        e.potential.max,
        e.gradient.l2,
        e.gradient.max,
    ]
}

fn row(rows: &[Row], chain: Chain, p: usize) -> &Row {
    rows.iter()
        .find(|r| r.chain == chain && r.p == p)
        .expect("row present")
}

#[test]
fn f64_sweep_is_complete_finite_and_converges() {
    let rows = sweep::<f64>(&SMOKE, P_MAX, Translations::Reference);
    assert_eq!(rows.len(), Chain::ALL.len() * P_MAX);
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.chain, Chain::ALL[i / P_MAX]);
        assert_eq!(r.p, i % P_MAX + 1);
        for v in values(&r.worst).into_iter().chain(values(&r.all)) {
            assert!(v.is_finite() && v > 0.0, "{r:?}");
            assert!(v < 1.0, "{r:?}");
        }
    }
    // The union over offsets can never exceed the worst single offset:
    // ‖e‖² / ‖φ*‖² summed over offsets is at most the largest per-offset ratio, and the
    // largest error over the union is at most its offset's ratio times the largest
    // reference.
    for r in &rows {
        for (all, worst) in values(&r.all).into_iter().zip(values(&r.worst)) {
            assert!(all <= worst * (1.0 + 1e-12), "{r:?}");
        }
        assert!(
            r.worst_offset.iter().any(|c| c.abs() > 1),
            "{r:?}: not a V-list offset"
        );
    }
    for chain in Chain::ALL {
        let (first, last) = (row(&rows, chain, 1), row(&rows, chain, P_MAX));
        for (a, b) in values(&first.all).into_iter().zip(values(&last.all)) {
            assert!(
                b < a,
                "{chain:?}: p = {P_MAX} error {b} not below p = 1 error {a}"
            );
        }
    }
}

#[test]
fn exact_translations_do_not_change_the_error() {
    // M2M and L2L are exact to degree p (CONVENTIONS §3.11), so the upward chain agrees
    // with P2M → M2P and the downward chain with P2M → M2L → L2P up to rounding.
    let rows = sweep::<f64>(&SMOKE, P_MAX, Translations::Reference);
    for (exact, with) in [
        (Chain::P2mM2p, Chain::P2mM2mM2p),
        (Chain::P2mM2lL2p, Chain::P2mM2lL2lL2p),
    ] {
        for p in 1..=P_MAX {
            let (a, b) = (row(&rows, exact, p), row(&rows, with, p));
            for (x, y) in values(&a.all)
                .into_iter()
                .chain(values(&a.worst))
                .zip(values(&b.all).into_iter().chain(values(&b.worst)))
            {
                assert!((x - y).abs() <= 1e-12, "{with:?}, p = {p}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn f32_sweep_matches_f64_where_truncation_dominates() {
    // At p ≤ 4 the truncation error (≳ 1e-4) is far above f32 rounding (≈ 1e-7), so
    // the f32 errors agree with the f64 ones to a few per cent.
    let single = sweep::<f32>(&SMOKE, P_MAX, Translations::Reference);
    let double = sweep::<f64>(&SMOKE, P_MAX, Translations::Reference);
    for (s, d) in single.iter().zip(&double) {
        assert_eq!((s.chain, s.p), (d.chain, d.p));
        for (x, y) in values(&s.all)
            .into_iter()
            .chain(values(&s.worst))
            .zip(values(&d.all).into_iter().chain(values(&d.worst)))
        {
            assert!((x - y).abs() <= 0.02 * y + 1e-5, "{s:?} vs {d:?}");
        }
    }
}

#[test]
fn deterministic_for_a_seed() {
    assert_eq!(
        sweep::<f64>(&SMOKE, 2, Translations::Reference),
        sweep::<f64>(&SMOKE, 2, Translations::Reference)
    );
    let other = Config { seed: 8, ..SMOKE };
    assert_ne!(
        sweep::<f64>(&SMOKE, 2, Translations::Reference),
        sweep::<f64>(&other, 2, Translations::Reference)
    );
}

#[test]
fn table_path_reproduces_the_reference_errors() {
    // The dense tables equal `nd_fmm_ref::direct` at their canonical frames up to
    // rounding (nd-fmm-tables, C2.1 and C2.2), and the sweep geometry is those frames
    // scaled by ½ (CONVENTIONS §3.7, §3.12), so in f64 every error figure agrees with
    // the reference path's to relative 1e-12. (In f32 the rounded tables and the f32
    // direct operators differ at f32 rounding, far above 1e-12.) Error measure:
    // |table − reference| / reference per figure.
    fn check(tables: &[Row], reference: &[Row], precision: &str) {
        assert_eq!(tables.len(), reference.len());
        for (t, r) in tables.iter().zip(reference) {
            assert_eq!((t.chain, t.p), (r.chain, r.p));
            for (x, y) in values(&t.all)
                .into_iter()
                .chain(values(&t.worst))
                .zip(values(&r.all).into_iter().chain(values(&r.worst)))
            {
                let rel = (x - y).abs() / y;
                assert!(rel <= 1e-12, "{precision} {t:?} vs {r:?}: relative {rel:e}");
            }
        }
    }
    let reference = sweep::<f64>(&SMOKE, P_MAX, Translations::Reference);
    let tables = sweep::<f64>(&SMOKE, P_MAX, Translations::Tables);
    check(&tables, &reference, "f64");
    // Chains without a translation run the same code on both paths.
    for (t, r) in tables.iter().zip(&reference) {
        if matches!(t.chain, Chain::P2mM2p | Chain::P2lL2p) {
            assert_eq!(t, r);
        }
    }
}

//! Translations, the C3.1 criterion for M2M, L2L and M2L: the per-pair kernels of
//! `LaplaceOperator` against `nd_fmm_ref::direct` at the absolute frames of
//! CONVENTIONS §3.12, on levels 2, 9 and 16 of the dyadic domain, for every octant and
//! every V-list offset (all 316 lie inside the domain on these levels).
//!
//! Error measure: output coefficients per degree in the orthonormal weighting, relative
//! to the terms |Aᵢₖ xₖ| of the dense f64 table (`common`), for every strategy. The
//! dyadic domain makes every centre and centre difference exact, so `direct` sees the
//! canonical shift and radius ratio exactly (asserted), and only the summation order
//! differs. Tolerances: 1e-14 with `Dense`, 1e-13 with `Classes` and `Rotation`.

use std::time::Instant;

use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::{m2l_offsets, octant_direction};
use nd_octree::morton;

use crate::common::{
    Dense, Kind, LEVELS, STRATEGIES, SplitMix64, Worst, absolute_frame, degree_error, dense,
    dyadic_domain, len, operator, random_coefficients, random_pair, terms, translation_tolerance,
};

/// The degrees of the debug run (C3.1: all offsets at p ≤ 6).
const DEBUG_DEGREES: [usize; 4] = [0, 1, 3, 6];

/// The worst errors of one strategy, per translation.
struct Worsts {
    m2m: Worst,
    l2l: Worst,
    m2l: Worst,
}

impl Worsts {
    fn new(strategy: M2lStrategy, degrees: &str) -> Self {
        let name = |op: &str| format!("{op} {strategy:?} vs direct, levels 2, 9, 16, {degrees}");
        Self {
            m2m: Worst::new(name("M2M")),
            l2l: Worst::new(name("L2L")),
            m2l: Worst::new(name("M2L")),
        }
    }
}

/// Checks M2M and L2L for every octant and M2L for every offset of `op` on every level
/// of `LEVELS`, `pairs` random pairs each, against `direct` with the terms of `dense`.
fn check_translations(
    op: &mut LaplaceOperator<f64>,
    dense: &Dense,
    pairs: usize,
    worst: &Worsts,
    rng: &mut SplitMix64,
) {
    let p = op.p();
    let strategy = op.tables().strategy();
    let tolerance = translation_tolerance(strategy);
    let domain = dyadic_domain();
    let mut ws = Workspace::new(p);
    for level in LEVELS {
        for o in 0..8 {
            for _ in 0..pairs {
                let parent = rng.key(level - 1);
                let child = morton::children(parent).unwrap()[o];
                let (from, to) = (
                    absolute_frame(child, &domain),
                    absolute_frame(parent, &domain),
                );
                // The canonical frames of §3.12, exactly.
                let half = octant_direction(o).map(|s| 0.5 * s as f64);
                assert_eq!(to.scaled(from.centre), half);
                assert_eq!(from.radius / to.radius, 0.5);
                let context = || format!("{strategy:?}, p = {p}, level {level}, octant {o}");

                let x = random_coefficients(Kind::Multipole, p, rng);
                let mut got = vec![0.0; len(p)];
                op.m2m_pair(child, parent, &x, &mut got);
                let mut want = vec![0.0; len(p)];
                direct::m2m(p, &from, &to, &mut ws, &x, &mut want);
                let e = degree_error(Kind::Multipole, p, &got, &want, &terms(&dense.m2m, o, &x));
                worst.m2m.check(e, tolerance, context);

                let x = random_coefficients(Kind::Local, p, rng);
                let mut got = vec![0.0; len(p)];
                op.l2l_pair(parent, child, &x, &mut got);
                let mut want = vec![0.0; len(p)];
                direct::l2l(p, &to, &from, &mut ws, &x, &mut want);
                let e = degree_error(Kind::Local, p, &got, &want, &terms(&dense.l2l, o, &x));
                worst.l2l.check(e, tolerance, context);
            }
        }
        for (index, d) in m2l_offsets().into_iter().enumerate() {
            for _ in 0..pairs {
                let (source, target) = random_pair(level, d, rng);
                let (from, to) = (
                    absolute_frame(source, &domain),
                    absolute_frame(target, &domain),
                );
                assert_eq!(from.scaled(to.centre), d.map(|t| 2.0 * t as f64));
                assert_eq!(to.radius, from.radius);
                let x = random_coefficients(Kind::Multipole, p, rng);
                let mut got = vec![0.0; len(p)];
                op.m2l_pair(source, target, &x, &mut got);
                let mut want = vec![0.0; len(p)];
                direct::m2l(p, &from, &to, &mut ws, &x, &mut want);
                let e = degree_error(Kind::Local, p, &got, &want, &terms(&dense.m2l, index, &x));
                worst.m2l.check(e, tolerance, || {
                    format!("{strategy:?}, p = {p}, level {level}, d = {d:?}")
                });
            }
        }
    }
}

fn check_strategy(strategy: M2lStrategy, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let worst = Worsts::new(strategy, "p ∈ {0, 1, 3, 6} (terms)");
    for p in DEBUG_DEGREES {
        let mut op = operator(strategy, p, 0);
        check_translations(&mut op, dense(p), 1, &worst, &mut rng);
    }
}

#[test]
fn dense_translations_equal_direct_on_levels_2_9_16() {
    check_strategy(M2lStrategy::Dense, 0x7801);
}

#[test]
fn class_translations_equal_direct_on_levels_2_9_16() {
    check_strategy(M2lStrategy::Classes, 0x7802);
}

#[test]
fn rotation_translations_equal_direct_on_levels_2_9_16() {
    check_strategy(M2lStrategy::Rotation, 0x7803);
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-exec --release -- --ignored`"]
fn translations_equal_direct_at_p_8_and_16() {
    // As the debug tests, at p = 8 for every strategy and p = 16 for `Rotation`, two
    // random pairs per octant or offset and level. The terms at p = 16 come from the
    // dense tables, built here only for the measure. Also reports the build time of
    // each strategy's tables at p = 8 and 16.
    let mut rng = SplitMix64::new(0x7804);
    for p in [8, 16] {
        for strategy in STRATEGIES {
            let start = Instant::now();
            let tables = Tables::<f64>::build(p, strategy);
            eprintln!(
                "build {strategy:?} p = {p}: {:.3?} ({:?})",
                start.elapsed(),
                tables.kinds()
            );
            if p == 16 && strategy != M2lStrategy::Rotation {
                continue;
            }
            let worst = Worsts::new(strategy, &format!("p = {p} (terms)"));
            let mut op = LaplaceOperator::new(tables, false, 0);
            check_translations(&mut op, dense(p), 2, &worst, &mut rng);
        }
    }
}

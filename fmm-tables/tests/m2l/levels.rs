//! Level independence (C2.2): a table built at the canonical frames reproduces
//! `direct::m2l` at the actual source and target frames of V-list pairs on levels 2, 9
//! and 16 of a dyadic domain, for every offset.

use std::time::Instant;

use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index, m2l_offsets};
use nd_fmm_tables::{M2lTables, MatrixSet};

use crate::common::{
    DEBUG_DEGREES, DYADIC, Kind, LEVEL_TOL, LEVELS, P_MAX_M2L, P_SUBSET, SplitMix64, Worst, apply,
    degree, degree_error, len, m2l, random_coefficients, random_pair, subset, terms,
};
use crate::tables::check_inversion;

/// Matrix `position` of `set`, the table of offset `d`, applied to a random multipole
/// against `direct::m2l` between the boxes `source` and `target` = source + d on
/// `level` of the dyadic domain, at the degree of `set`.
///
/// The dyadic domain makes the shift exact: the test asserts that `direct` sees
/// b = (c′ − c)/r = 2d and σ = 1 bit for bit, so only the summation order differs.
/// Returns the per-degree error relative to the terms of the table application.
pub fn check_pair(
    level: u32,
    (source, target): ([u64; 3], [u64; 3]),
    d: [i64; 3],
    set: &MatrixSet<f64>,
    position: usize,
    ws: &mut Workspace<f64>,
    rng: &mut SplitMix64,
) -> f64 {
    let p = degree(set);
    let source = DYADIC.frame(level, source);
    let target = DYADIC.frame(level, target);
    assert_eq!(source.scaled(target.centre), d.map(|t| 2.0 * t as f64));
    assert_eq!(target.radius / source.radius, 1.0);

    let input = random_coefficients(Kind::Multipole, p, rng);
    let got = apply(set, position, &input);
    let mut want = vec![0.0; len(p)];
    direct::m2l(p, &source, &target, ws, &input, &mut want);
    let scale = terms(set, position, &input);
    degree_error(Kind::Local, p, &got, &want, &scale)
}

/// Runs [`check_pair`] for every offset of `offsets` (with its position in `set`) on
/// every level of `LEVELS`, at `pairs` random pairs each, and asserts `LEVEL_TOL`.
fn check_levels(
    set: &MatrixSet<f64>,
    offsets: &[([i64; 3], usize)],
    pairs: usize,
    worst: &Worst,
    rng: &mut SplitMix64,
) {
    let p = degree(set);
    let mut ws = Workspace::new(p);
    for level in LEVELS {
        for &(d, position) in offsets {
            for _ in 0..pairs {
                let pair = random_pair(level, d, rng);
                let e = worst.update(check_pair(level, pair, d, set, position, &mut ws, rng));
                assert!(
                    e <= LEVEL_TOL,
                    "level {level}, pair {pair:?}, d = {d:?}, p = {p}: {e:.3e}"
                );
            }
        }
    }
}

/// Every offset with its table index, the position of its matrix in a full table.
fn all_offsets() -> Vec<([i64; 3], usize)> {
    m2l_offsets().into_iter().zip(0..M2L_OFFSET_COUNT).collect()
}

#[test]
fn tables_equal_direct_on_levels_2_9_16_of_a_dyadic_domain() {
    // Error measure: local coefficients per degree in the orthonormal weighting
    // Nₘ/Sₘ, relative to the terms |Aᵢₖ xₖ| of the table application (`support`). The
    // dyadic domain a = (−1.25, 0.5, 2), w = 3 makes every centre and centre
    // difference exact, so `direct` sees b = 2d and σ = 1 exactly. All 316 offsets,
    // two random pairs per offset and level, p ≤ 8.
    let mut rng = SplitMix64::new(0x4d21_0001);
    let worst = Worst::new("M2L table vs direct, levels 2, 9, 16, p ≤ 8, per degree (terms)");
    for p in DEBUG_DEGREES {
        check_levels(m2l(p).matrices(), &all_offsets(), 2, &worst, &mut rng);
    }
}

#[test]
fn class_representatives_equal_direct_on_levels_2_9_16_at_p_12() {
    // Error measure: as `tables_equal_direct_on_levels_2_9_16_of_a_dyadic_domain`, for
    // the 16 class representatives of CONVENTIONS §3.12 at p = 12, four random pairs
    // per representative and level.
    let mut rng = SplitMix64::new(0x4d21_0002);
    let worst = Worst::new("M2L table vs direct, levels 2, 9, 16, class representatives, p = 12");
    let subset = subset();
    let offsets: Vec<([i64; 3], usize)> = crate::common::CLASS_REPRESENTATIVES
        .iter()
        .map(|&d| (d, subset.position(m2l_offset_index(d).unwrap())))
        .collect();
    assert_eq!(degree(&subset.matrices), P_SUBSET);
    check_levels(&subset.matrices, &offsets, 4, &worst, &mut rng);
}

#[test]
#[ignore = "large p: run with `cargo test -p nd-fmm-tables --release -- --ignored`"]
fn tables_equal_direct_on_levels_2_9_16_at_large_p() {
    // As `tables_equal_direct_on_levels_2_9_16_of_a_dyadic_domain`, for all 316
    // offsets at p = 12, 16 and 20. Also reports the build time and the storage of the
    // table at p = 8, 12, 16 and 20 (the peak memory of a build is the table plus
    // O((p + 1)²) scratch), and checks the inversion identity at p = 16 and 20.
    for p in [8, 12, 16, P_MAX_M2L] {
        let start = Instant::now();
        let tables = M2lTables::<f64>::build(p);
        let elapsed = start.elapsed();
        let bytes = core::mem::size_of_val(tables.matrices().as_slice());
        eprintln!(
            "build p = {p}: {elapsed:.3?}, storage {bytes} bytes ({:.1} MB)",
            bytes as f64 / 1e6
        );
        if p > 8 {
            let worst = Worst::new(match p {
                12 => "M2L table vs direct, levels 2, 9, 16, p = 12, per degree (terms)",
                16 => "M2L table vs direct, levels 2, 9, 16, p = 16, per degree (terms)",
                _ => "M2L table vs direct, levels 2, 9, 16, p = 20, per degree (terms)",
            });
            let mut rng = SplitMix64::new(0x4d21_0003 + p as u64);
            check_levels(tables.matrices(), &all_offsets(), 2, &worst, &mut rng);
        }
        if p >= 16 {
            check_inversion(tables.matrices()).print(match p {
                16 => "inversion p = 16",
                _ => "inversion p = 20",
            });
        }
    }
}

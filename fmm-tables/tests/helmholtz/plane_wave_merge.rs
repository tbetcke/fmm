//! Merge and shift's tables of the Helmholtz exponential M2L (Helmholtz CH6.1d,
//! `nd_fmm_tables::helmholtz::plane_wave::merge`; CONVENTIONS §3.14, "Merge and shift",
//! with §3.15's shifts): the slot and child shifts against the pair's shift, and the
//! choice of a level's quadrature.
//!
//! Error measure: per wave value, |S_slot S_child − T_pair| relative to |S_slot| |S_child|
//! (the size of the product the merge forms), the largest over every direction, slot of
//! an accumulator and child of its mask.

use nd_fmm_math::Complex;
use nd_fmm_tables::helmholtz::plane_wave::merge::{
    MAX_PLANE_WAVE_DEGREE, MergeLevel, PLANE_WAVE_MIN_TOLERANCE, PlaneWaveM2l, RULE_PRECISION, add,
    precision_for_degree, precision_for_tolerance, select, shift,
};
use nd_fmm_tables::helmholtz::plane_wave::pair::shift_add;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{RHO_MAX, domain_of};
use nd_fmm_tables::helmholtz::plane_wave::{Precision, Quadrature};
use nd_fmm_tables::plane_wave::geometry::{
    CHILDREN, Direction, MergeGeometry, child_bits, slot_coordinates,
};

type C = Complex<f64>;

/// The bound of the factorisation, relative to the product's size: the two factors and
/// the pair's are each formed in f64 from exp of an argument up to about 100 in modulus
/// (7.5e-14 measured on the M6; a tenfold margin for other libms).
const FACTOR_BOUND: f64 = 1e-12;

/// The largest relative difference of slot shift × child shift from the pair's shift over
/// every direction, accumulator slot and child of its mask; also checks each slot's shift
/// bit for bit against `shift_factors` of its own offset.
fn check_level(level: &MergeLevel<f64>, geometry: &MergeGeometry) -> f64 {
    let tables = level.tables();
    let w = tables.wave_len() / 2;
    let mut pair = vec![0.0; 2 * w];
    let mut own = vec![0.0; 2 * w];
    let mut worst = 0.0f64;
    for direction in Direction::ALL {
        let up = direction.is_up();
        for accumulator in geometry.accumulators(direction) {
            for &slot in &accumulator.slots {
                let a = slot_coordinates(slot as usize);
                let source = level
                    .source_shift(direction, slot as usize)
                    .expect("a slot of an accumulator has a shift");
                tables.shift_factors(
                    direction.rotate(a.map(|x| (5 - 2 * x) as f64 / 2.0)),
                    up,
                    &mut own,
                );
                assert!(
                    source.iter().zip(&own).all(|(x, y)| x == y),
                    "{direction:?}, slot {slot}: the shared shift is not the slot's own"
                );
                for child in (0..CHILDREN).filter(|c| accumulator.mask & (1 << c) != 0) {
                    let b = child_bits(child);
                    let d: [f64; 3] = core::array::from_fn(|i| (b[i] + 2 - a[i]) as f64);
                    tables.shift_factors(direction.rotate(d), up, &mut pair);
                    let child_shift = level.child_shift(direction, child);
                    for j in 0..w {
                        let s = C::new(source[j], source[w + j]);
                        let c = C::new(child_shift[j], child_shift[w + j]);
                        let t = C::new(pair[j], pair[w + j]);
                        let size = s.norm() * c.norm();
                        if size > 0.0 {
                            worst = worst.max((s * c - t).norm() / size);
                        }
                    }
                }
            }
        }
    }
    worst
}

/// k̂₀ (the root's k r₀) and the levels checked: real, complex and strongly damped k, so
/// that κ_l = 2k̂₀ 2⁻ˡ meets both node maps (Im κ below and above π/2).
const CASES: [((f64, f64), &[usize]); 3] = [
    ((80.0, 0.0), &[2, 5, 9]),
    ((80.0, 12.0), &[3, 6]),
    ((20.0, 40.0), &[2, 4, 8]),
];

#[test]
fn slot_and_child_shifts_factor_the_pair_shift() {
    let geometry = MergeGeometry::new();
    let debug = cfg!(debug_assertions);
    let mut worst = 0.0f64;
    for (i, ((re, im), levels)) in CASES.into_iter().enumerate() {
        for (j, precision) in Precision::ALL.into_iter().enumerate() {
            for (n, &l) in levels.iter().enumerate() {
                // Debug: one level per case and precision, in turn.
                if debug && n != (i + j) % levels.len() {
                    continue;
                }
                let k_hat = C::new(re, im) * 0.5f64.powi(l as i32);
                let level = MergeLevel::<f64>::new(k_hat, 8, precision, &geometry)
                    .expect("κ lies in the tables");
                let error = check_level(&level, &geometry);
                println!(
                    "k̂₀ = {re} + {im}i, level {l}, iprec {}: {} source shifts, worst {error:.2e}",
                    precision.iprec(),
                    level.source_shift_count()
                );
                worst = worst.max(error);
            }
        }
    }
    assert!(worst <= FACTOR_BOUND, "worst {worst:e}");
}

#[test]
fn the_rule_pairs_tolerances_with_precisions() {
    let iprec2 = Some(Precision::Iprec2);
    assert_eq!(RULE_PRECISION, Precision::Iprec2);
    for (eps, want) in [
        (1e-1, iprec2),
        (1e-3, iprec2),
        (1e-6, iprec2),
        (PLANE_WAVE_MIN_TOLERANCE, iprec2),
        (PLANE_WAVE_MIN_TOLERANCE * 0.999, None),
        (1e-7, None),
        (1e-8, None),
        (0.0, None),
        (-1e-3, None),
        (f64::NAN, None),
        (f64::INFINITY, None),
    ] {
        assert_eq!(precision_for_tolerance(eps), want, "ε = {eps:e}");
    }
}

#[test]
fn the_rule_pairs_degrees_with_precisions() {
    for p in 0..=40 {
        let want = (p <= MAX_PLANE_WAVE_DEGREE).then_some(Precision::Iprec2);
        assert_eq!(precision_for_degree(p), want, "p = {p}");
    }
    // The calibrated plateau ⌈0.30 L² + 0.25 L + 2.60⌉ at the rule's tolerance
    // (`nd_fmm_exec::helmholtz::DegreeRule::CALIBRATED`; its own test checks the same).
    let l = (1.0 / PLANE_WAVE_MIN_TOLERANCE).log10();
    let plateau = ((0.30 * l + 0.25) * l + 2.60 - 1e-9).ceil() as usize;
    assert_eq!(MAX_PLANE_WAVE_DEGREE, plateau);
    assert_eq!(MAX_PLANE_WAVE_DEGREE, 15);
}

#[test]
fn select_applies_the_rule_where_the_tables_reach() {
    let iprec2 = Some(Precision::Iprec2);
    // κ₀ = 2k̂₀ = 400: levels 0–2 (κ_l = 400, 200, 100 ≥ 16π) lie outside the tables.
    let k_root = C::new(200.0, 0.0);
    let degrees = [30, 30, 30, 30, 30, 24, 15, 8, 6];
    for tolerance in [None, Some(1e-3), Some(1e-6), Some(1e-8)] {
        let chosen = select(k_root, &degrees, tolerance);
        for (l, (&p, &got)) in degrees.iter().zip(&chosen).enumerate() {
            let kappa = k_root * 2.0 * 0.5f64.powi(l as i32);
            let want = if l >= 2 && domain_of(kappa).is_some() {
                tolerance.map_or(precision_for_degree(p), precision_for_tolerance)
            } else {
                None
            };
            assert_eq!(got, want, "level {l}, tolerance {tolerance:?}");
        }
        assert_eq!(&chosen[..3], &[None; 3]);
    }
    // By the degrees: levels 3–5 above the rule's largest degree.
    assert_eq!(
        &select(k_root, &degrees, None)[3..],
        &[None, None, None, iprec2, iprec2, iprec2]
    );
    // By the tolerance, whatever the degrees.
    assert_eq!(&select(k_root, &degrees, Some(1e-3))[3..], &[iprec2; 6]);
    assert_eq!(&select(k_root, &degrees, Some(1e-6))[3..], &[iprec2; 6]);
    assert_eq!(&select(k_root, &degrees, Some(1e-8))[3..], &[None; 6]);
    // Complex k; and k̂₀ with Re k̂₀ = 0 (Yukawa) or Im k̂₀ < 0, none.
    assert_eq!(
        select(C::new(3.0, 1.0), &[8, 8, 8, 8], None),
        [None, None, iprec2, iprec2]
    );
    for tolerance in [None, Some(1e-6)] {
        assert_eq!(select(C::new(0.0, 3.0), &[8; 4], tolerance), [None; 4]);
        assert_eq!(select(C::new(3.0, -1.0), &[8; 4], tolerance), [None; 4]);
    }
}

#[test]
fn build_takes_named_precisions() {
    let k_root = C::new(200.0, 0.0);
    let degrees = [8; 6];
    // Level 2 (κ = 100) lies outside the tables: no quadrature there.
    let named = [
        None,
        None,
        Some(Precision::Iprec3),
        Some(Precision::Iprec3),
        None,
        Some(Precision::Iprec2),
    ];
    let m2l = PlaneWaveM2l::<f64>::build(k_root, &degrees, &named);
    assert_eq!(m2l.nlevels(), 6);
    assert_eq!(
        (0..6).map(|l| m2l.precision(l)).collect::<Vec<_>>(),
        [
            None,
            None,
            None,
            Some(Precision::Iprec3),
            None,
            Some(Precision::Iprec2)
        ]
    );
    let level = m2l.level(3).expect("level 3 has tables");
    assert_eq!(level.tables().k_hat(), k_root * 0.125);
    assert_eq!(level.tables().p(), 8);
    assert_eq!(m2l.bytes(), level.bytes() + m2l.level(5).unwrap().bytes());
    assert!(level.bytes() > level.tables().bytes());
    let none = PlaneWaveM2l::<f64>::build(k_root, &degrees, &[None; 6]);
    assert_eq!(none.bytes(), 0);
}

/// The levels' shifts, every slot's and child's, in one list (bit for bit comparisons).
fn shifts(level: &MergeLevel<f64>, geometry: &MergeGeometry) -> Vec<f64> {
    let mut out = Vec::new();
    for direction in Direction::ALL {
        for accumulator in geometry.accumulators(direction) {
            for &slot in &accumulator.slots {
                out.extend_from_slice(level.source_shift(direction, slot as usize).unwrap());
            }
        }
        for child in 0..CHILDREN {
            out.extend_from_slice(level.child_shift(direction, child));
        }
    }
    out
}

/// The counts for a smaller reach (Helmholtz CH6.2e): `build_with_reach` at the corner's
/// reach is `build` bit for bit; at FMM3D's reach 4 every level has at most the corner's
/// waves, its quadrature's counts those of `Quadrature::with_reach`.
#[test]
fn build_with_reach_takes_the_counts_of_the_reach() {
    let k_root = C::new(60.0, 0.5);
    let degrees = [8, 8, 9, 10, 11];
    let named = [
        None,
        None,
        Some(Precision::Iprec2),
        Some(Precision::Iprec2),
        Some(Precision::Iprec1),
    ];
    let corner = PlaneWaveM2l::<f64>::build(k_root, &degrees, &named);
    let same = PlaneWaveM2l::<f64>::build_with_reach(k_root, &degrees, &named, RHO_MAX);
    let fewer = PlaneWaveM2l::<f64>::build_with_reach(k_root, &degrees, &named, 4.0);
    let geometry = corner.geometry();
    for l in 2..5 {
        let (a, b, c) = (
            corner.level(l).unwrap(),
            same.level(l).unwrap(),
            fewer.level(l).unwrap(),
        );
        assert_eq!(a.tables().wave_len(), b.tables().wave_len());
        let (sa, sb) = (shifts(a, geometry), shifts(b, geometry));
        assert!(sa.iter().zip(&sb).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(c.tables().wave_len() <= a.tables().wave_len(), "level {l}");
        let kappa = k_root * (0.5f64).powi(l as i32) * 2.0;
        let q = Quadrature::with_reach(kappa, named[l].unwrap(), 4.0).unwrap();
        assert_eq!(c.tables().wave_len(), 2 * q.wave_count(), "level {l}");
    }
    assert!(fewer.bytes() < corner.bytes());
}

#[test]
#[should_panic(expected = "levels 0 and 1")]
fn build_refuses_a_precision_on_level_one() {
    let _ = PlaneWaveM2l::<f64>::build(
        C::new(1.0, 0.0),
        &[4, 4, 4],
        &[None, Some(Precision::Iprec1), None],
    );
}

#[test]
#[should_panic(expected = "one precision per level")]
fn build_refuses_a_short_precision_list() {
    let _ = PlaneWaveM2l::<f64>::build(C::new(1.0, 0.0), &[4, 4, 4], &[None, None]);
}

#[test]
fn shift_and_add_are_the_complex_steps() {
    let v: Vec<f64> = (0..8).map(|i| 0.3 + 0.17 * i as f64).collect();
    let f: Vec<f64> = (0..8).map(|i| 1.1 - 0.23 * i as f64).collect();
    let mut shifted = v.clone();
    shift(&mut shifted, &f);
    let mut from_zero = vec![0.0; 8];
    shift_add(&mut from_zero, &v, &f);
    assert_eq!(
        shifted, from_zero,
        "shift is shift_add from zero, bit for bit"
    );
    let mut sum = v.clone();
    add(&mut sum, &f);
    for i in 0..8 {
        assert_eq!(sum[i], v[i] + f[i]);
    }
}

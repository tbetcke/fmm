//! The leaf rule (proposal §3.7): `min_leaf_level` against the census spike's search
//! (`spikes/census/src/tree.rs`, `width_level`, the same inequality) and against the
//! closed form ⌈log₂(k w / (2π c_leaf))⌉ (an independent path, away from the level
//! boundaries), its minimality, known levels, complex and Yukawa k, the cap and the
//! domain's scaling; and the points per leaf of Helmholtz CH4.0d3
//! (`default_max_points_per_leaf`, fmm-validate/results/helmholtz-ch4-0d3.md): its table, a
//! function of the tolerance and the ISA only, never fewer points at a smaller tolerance,
//! and `DEFAULT_C_LEAF`.

use std::f64::consts::{PI, TAU};

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    C_LEAF, DEFAULT_C_LEAF, Wavenumber, default_max_points_per_leaf, min_leaf_level,
};
use nd_fmm_exec::operator::Isa;
use nd_octree::PhysicalBox;

/// The trusted path: the census spike's `width_level`, the shallowest level whose boxes
/// are at most c_leaf λ wide in the root cube of side 1, λ = 2π/k.
fn width_level(c_leaf: f64, k: f64) -> usize {
    let lambda = 2.0 * PI / k;
    (0..=16)
        .find(|&l| 0.5f64.powi(l) <= c_leaf * lambda)
        .map_or(16, |l| l as usize)
}

fn cube(lower: f64, side: f64) -> Domain {
    Domain::new(&PhysicalBox::new([
        lower,
        lower,
        lower,
        lower + side,
        lower + side,
        lower + side,
    ]))
    .unwrap()
}

/// kD from 10⁻² to 10⁵, 40 per decade, and the powers of two near the boundaries.
fn kd_sweep() -> Vec<f64> {
    let mut kds: Vec<f64> = (0..=280)
        .map(|i| 10f64.powf(-2.0 + i as f64 / 40.0))
        .collect();
    for l in 0..=17 {
        // k = 2π c 2ˡ puts a level boundary exactly at c = 1.
        let at = TAU * 2f64.powi(l);
        kds.extend([at, at.next_up(), at.next_down()]);
    }
    kds
}

#[test]
fn level_matches_the_census_search() {
    let domain = cube(0.0, 1.0);
    let mut checked = 0;
    for c_leaf in [0.25, 0.5, 1.0, 1.5, 2.0] {
        for kd in kd_sweep() {
            let k = Wavenumber::new(kd, 0.0).unwrap();
            assert_eq!(
                min_leaf_level(&k, &domain, c_leaf),
                width_level(c_leaf, kd),
                "kD = {kd}, c_leaf = {c_leaf}"
            );
            checked += 1;
        }
    }
    println!("{checked} (kD, c_leaf) pairs equal to the census search");
}

#[test]
fn level_matches_the_closed_form() {
    // w 2⁻ˡ ≤ c λ ⟺ l ≥ log₂(k w / (2π c)): l = max(0, ⌈log₂(k w / (2π c))⌉), capped at 16,
    // computed in another way than the search; away from the boundaries, where the
    // logarithm's rounding decides.
    let domain = cube(0.0, 1.0);
    let mut checked = 0;
    for c_leaf in [0.25, 0.5, 1.0, 1.5, 2.0] {
        for kd in kd_sweep() {
            let x = (kd / (TAU * c_leaf)).log2();
            if (x - x.round()).abs() < 1e-9 {
                continue;
            }
            let expected = (x.ceil().max(0.0) as usize).min(16);
            let k = Wavenumber::new(kd, 0.0).unwrap();
            assert_eq!(
                min_leaf_level(&k, &domain, c_leaf),
                expected,
                "kD = {kd}, c_leaf = {c_leaf}"
            );
            checked += 1;
        }
    }
    println!("{checked} (kD, c_leaf) pairs equal to the closed form");
}

#[test]
fn level_is_the_shallowest_narrow_enough() {
    for (lower, side) in [(0.0, 1.0), (-1.0, 2.0), (3.5, 0.125), (-1e3, 7.3)] {
        let domain = cube(lower, side);
        for kd in kd_sweep() {
            let k = Wavenumber::new(kd / side, 0.0).unwrap();
            let lambda = k.wavelength().unwrap();
            let l = min_leaf_level(&k, &domain, C_LEAF);
            let width = |l: usize| domain.side() * 0.5f64.powi(l as i32);
            if l < 16 {
                assert!(width(l) <= C_LEAF * lambda, "kD = {kd}, level {l}");
            }
            if l > 0 {
                assert!(width(l - 1) > C_LEAF * lambda, "kD = {kd}, level {l}");
            }
        }
    }
}

#[test]
fn known_levels_of_the_reference_range() {
    // D = 1, c_leaf = 1 (D11): λ = 2π/kD, and the leaves of the census (SPIKE_REPORT.md).
    let domain = cube(0.0, 1.0);
    for (kd, level) in [
        (1.0, 0),
        (6.0, 0),
        (10.0, 1),
        (30.0, 3),
        (100.0, 4),
        (200.0, 5),
        (300.0, 6),
        (500.0, 7),
        (1000.0, 8),
    ] {
        let k = Wavenumber::new(kd, 0.0).unwrap();
        assert_eq!(min_leaf_level(&k, &domain, C_LEAF), level, "kD = {kd}");
    }
}

#[test]
fn complex_k_takes_the_level_of_re_k() {
    let domain = cube(0.0, 1.0);
    for kd in kd_sweep() {
        let real = Wavenumber::new(kd, 0.0).unwrap();
        for im in [1e-3 * kd, 0.1 * kd, kd, 10.0 * kd] {
            let complex = Wavenumber::new(kd, im).unwrap();
            assert_eq!(
                min_leaf_level(&complex, &domain, C_LEAF),
                min_leaf_level(&real, &domain, C_LEAF),
                "kD = {kd} + {im}i"
            );
        }
    }
}

#[test]
fn yukawa_has_no_rule() {
    let domain = cube(0.0, 1.0);
    for kappa in [1e-3, 1.0, 500.0, 1e6] {
        let k = Wavenumber::new(0.0, kappa).unwrap();
        assert_eq!(min_leaf_level(&k, &domain, C_LEAF), 0, "κ = {kappa}");
    }
}

#[test]
fn level_is_capped_at_16() {
    let domain = cube(0.0, 1.0);
    // 2⁻¹⁶ > c λ for kD > 2π 2¹⁶ ≈ 4.1e5.
    for kd in [5e5, 1e8, 1e300] {
        let k = Wavenumber::new(kd, 0.0).unwrap();
        assert_eq!(min_leaf_level(&k, &domain, C_LEAF), 16, "kD = {kd}");
    }
}

#[test]
fn level_depends_on_k_times_the_side() {
    // Powers of two keep k w exact, so the level is the same bit for bit.
    for kd in kd_sweep() {
        let reference = min_leaf_level(&Wavenumber::new(kd, 0.0).unwrap(), &cube(0.0, 1.0), 1.0);
        for s in [-3, -1, 1, 4] {
            let side = 2f64.powi(s);
            let k = Wavenumber::new(kd / side, 0.0).unwrap();
            assert_eq!(
                min_leaf_level(&k, &cube(-0.5 * side, side), 1.0),
                reference,
                "kD = {kd}, side {side}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "c_leaf")]
fn zero_c_leaf_panics() {
    let k = Wavenumber::new(1.0, 0.0).unwrap();
    min_leaf_level(&k, &cube(0.0, 1.0), 0.0);
}

#[test]
#[should_panic(expected = "c_leaf")]
fn negative_c_leaf_panics() {
    let k = Wavenumber::new(1.0, 0.0).unwrap();
    min_leaf_level(&k, &cube(0.0, 1.0), -1.0);
}

#[test]
#[should_panic(expected = "c_leaf")]
fn infinite_c_leaf_panics() {
    let k = Wavenumber::new(1.0, 0.0).unwrap();
    min_leaf_level(&k, &cube(0.0, 1.0), f64::INFINITY);
}

#[test]
#[should_panic(expected = "c_leaf")]
fn nan_c_leaf_panics() {
    let k = Wavenumber::new(1.0, 0.0).unwrap();
    min_leaf_level(&k, &cube(0.0, 1.0), f64::NAN);
}

/// The measured table: x86 (and the scalar kernel) 256 at every tolerance, NEON 128 at
/// ε ≥ 10⁻⁷ or named degrees and 256 below.
#[test]
fn the_points_per_leaf_by_the_tolerance_and_the_isa() {
    for isa in Isa::all() {
        for (epsilon, neon) in [
            (1e-3, 128),
            (1e-6, 128),
            (1e-7, 128),
            (9e-8, 256),
            (1e-8, 256),
        ] {
            let expected = if isa == Isa::Neon { neon } else { 256 };
            assert_eq!(
                default_max_points_per_leaf(Some(epsilon), isa),
                expected,
                "{isa:?}, ε = {epsilon:e}"
            );
        }
        let named = if isa == Isa::Neon { 128 } else { 256 };
        assert_eq!(
            default_max_points_per_leaf(None, isa),
            named,
            "{isa:?}, named"
        );
    }
}

#[test]
fn the_points_per_leaf_never_fall_with_the_tolerance() {
    for isa in Isa::all() {
        let sizes: Vec<usize> = (1..=12)
            .map(|e| default_max_points_per_leaf(Some(10f64.powi(-e)), isa))
            .collect();
        assert!(sizes.windows(2).all(|w| w[0] <= w[1]), "{isa:?}: {sizes:?}");
        assert!(sizes.iter().all(|&n| n >= 64), "{isa:?}: {sizes:?}");
    }
}

#[test]
fn the_default_leaf_width_is_two_wavelengths_one_level_coarser() {
    assert_eq!(DEFAULT_C_LEAF, 2.0 * C_LEAF);
    for kd in kd_sweep() {
        let k = Wavenumber::new(kd, 0.0).unwrap();
        let narrow = min_leaf_level(&k, &cube(0.0, 1.0), C_LEAF);
        let wide = min_leaf_level(&k, &cube(0.0, 1.0), DEFAULT_C_LEAF);
        // One level coarser, but never above the root, and at the cap both may hold 16.
        assert!(
            wide == narrow.saturating_sub(1) || (narrow == 16 && wide == 16),
            "kD = {kd}: {wide} against {narrow}"
        );
    }
}

//! `centre`, `radius` and `integer_centre` against the midpoint and half-side of
//! `morton::physical_box` (CONVENTIONS §3.12, "Domain and levels"; §3.13, "Integer
//! centres"), on levels 0–16.

use nd_fmm_exec::geometry::{centre, integer_centre, radius};
use nd_octree::morton;

use crate::common::{
    DEEPEST, EPS64, SplitMix64, Worst, dyadic_domain, generic_domain, physical_midpoint,
    sample_keys,
};

/// Tolerance of `centre` and `radius` against `physical_box` in the generic domain, in
/// units of ε₆₄ (|a_k| + w) per component. Both sides round a few times, and
/// `physical_box` uses the side w_k of its axis, which differs from w by about
/// ε₆₄ (|a| + w).
const GENERIC_ULPS: f64 = 8.0;

#[test]
fn centre_and_radius_are_exact_in_the_dyadic_domain() {
    // Error measure: bit-for-bit equality.
    let mut rng = SplitMix64::new(0x7304);
    let domain = dyadic_domain();
    let mut checked = 0;
    for level in 0..=DEEPEST {
        for key in sample_keys(&mut rng, level, 30) {
            let (mid, half) = physical_midpoint(key, &domain);
            assert_eq!(centre(key, &domain), mid, "key {key} on level {level}");
            assert_eq!(half, [radius(level, &domain); 3], "level {level}");
            checked += 1;
        }
    }
    println!("dyadic domain: centre and radius exact for {checked} keys on levels 0-16");
}

#[test]
fn centre_and_radius_are_within_a_few_ulps_in_the_generic_domain() {
    // Error measure: componentwise absolute error against the midpoint and half-side of
    // `physical_box`, in units of ε₆₄ (|a_k| + w).
    let mut rng = SplitMix64::new(0x7305);
    let domain = generic_domain();
    let (a, w) = (domain.lower(), domain.side());
    let mut worst_centre =
        Worst::new("generic domain: centre vs physical_box midpoint / ε₆₄ (|a| + w)");
    let mut worst_radius =
        Worst::new("generic domain: radius vs physical_box half-side / ε₆₄ (|a| + w)");
    for level in 0..=DEEPEST {
        for key in sample_keys(&mut rng, level, 30) {
            let (mid, half) = physical_midpoint(key, &domain);
            let c = centre(key, &domain);
            let r = radius(level, &domain);
            for k in 0..3 {
                let unit = EPS64 * (a[k].abs() + w);
                worst_centre.record((c[k] - mid[k]).abs() / unit);
                worst_radius.record((r - half[k]).abs() / unit);
            }
        }
    }
    worst_centre.check(GENERIC_ULPS);
    worst_radius.check(GENERIC_ULPS);
}

#[test]
fn integer_centres_reproduce_the_centre_in_the_dyadic_domain() {
    // Error measure: bit-for-bit equality of a + C_L r_L with `centre`, on every level
    // and every reference level L ≥ l.
    let mut rng = SplitMix64::new(0x7306);
    let domain = dyadic_domain();
    let a = domain.lower();
    let mut checked = 0;
    for level in 0..=DEEPEST {
        for key in sample_keys(&mut rng, level, 12) {
            let want = centre(key, &domain);
            for reference in level..=DEEPEST {
                let c = integer_centre(key, reference);
                let r = radius(reference, &domain);
                let got: [f64; 3] = core::array::from_fn(|k| a[k] + c[k] as f64 * r);
                assert_eq!(got, want, "key {key}, l = {level}, L = {reference}");
                // An odd multiple of 2^(L − l) in [1, 2^(L+1) − 1].
                let step = 1i64 << (reference - level);
                for ck in c {
                    assert!(ck % step == 0 && (ck / step) % 2 == 1, "C_L = {c:?}");
                    assert!((1..(1i64 << (reference + 1))).contains(&ck), "C_L = {c:?}");
                }
                checked += 1;
            }
        }
    }
    println!("integer centres: {checked} (key, L) pairs exact on levels 0-16, all L >= l");
}

#[test]
#[should_panic(expected = "reference level 2 must lie between the key's level 3 and 16")]
fn integer_centre_rejects_a_coarser_reference_level() {
    let _ = integer_centre(morton::from_index_and_level([1, 2, 3], 3), 2);
}

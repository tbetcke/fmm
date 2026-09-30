//! Round trips: `store` then `load`, and `load_or_build` cold and warm, against the
//! cold build.

use std::fmt::Debug;

use nd_fmm_tables::cache::{CacheOutcome, CachedTable};

use crate::common::{DEGREES, TestDir, assert_bit_identical, for_each_family};

/// Stores the build of `F` at every degree of [`DEGREES`] and loads it back.
fn check_round_trip<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    for p in DEGREES {
        let built = F::build(p);
        cache.store(&built).unwrap();
        let loaded = cache.load::<F>(p).unwrap();
        assert_eq!(loaded.p(), p);
        assert_bit_identical(dir, &loaded, &built);
    }
}

#[test]
fn store_then_load_is_bit_identical_for_every_family() {
    // Error measure: bit-for-bit equality with the cold build (PartialEq on every
    // field, and byte-identical files, whose payload holds every value's bits).
    let dir = TestDir::new("round-trip");
    for_each_family!(check_round_trip(&dir));
}

/// `load_or_build` of `F` at p = 1: `Built` when cold, then `Loaded`.
fn check_load_or_build<F: CachedTable + PartialEq + Debug>(dir: &TestDir) {
    let cache = dir.cache("cache");
    let p = 1;
    let cold = F::build(p);
    let (first, outcome) = cache.load_or_build::<F>(p);
    assert!(matches!(outcome, CacheOutcome::Built), "{outcome:?}");
    assert_bit_identical(dir, &first, &cold);
    let (second, outcome) = cache.load_or_build::<F>(p);
    assert!(matches!(outcome, CacheOutcome::Loaded), "{outcome:?}");
    assert_bit_identical(dir, &second, &cold);
}

#[test]
fn load_or_build_builds_when_cold_then_loads() {
    // Error measure: bit-for-bit equality with the cold build.
    let dir = TestDir::new("load-or-build");
    for_each_family!(check_load_or_build(&dir));
}

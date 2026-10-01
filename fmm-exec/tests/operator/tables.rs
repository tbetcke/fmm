//! Tables: `Tables::load_or_build` with a cache directory under the target's temporary
//! directory reports `Built`, then `Loaded`, for every family, with results
//! bit-identical to `Tables::build` without a cache. p ≤ 4 in the debug run; an
//! ignored release test reports the build and load times at p = 8 and 16.

use std::path::PathBuf;
use std::time::Instant;

use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;
use nd_fmm_tables::{CacheOutcome, TableCache};

use crate::common::{STRATEGIES, SplitMix64, len};

/// An empty cache directory for one test, under `CARGO_TARGET_TMPDIR`.
fn empty_cache(name: &str) -> TableCache {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("nd-fmm-exec-{name}"));
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => panic!("cannot clear {}: {error}", dir.display()),
    }
    TableCache::new(dir)
}

/// The outputs of every translation of `tables` for one random input each, as bits.
fn outputs<T: RealScalar>(tables: &Tables<T>, seed: u64) -> Vec<u64> {
    let mut rng = SplitMix64::new(seed);
    let n = len(tables.p());
    let mut scratch = tables.scratch();
    let mut bits = Vec::new();
    let mut record = |y: &[T]| bits.extend(y.iter().map(|&v| RealScalar::to_f64(v).to_bits()));
    for i in 0..316 {
        let x: Vec<T> = (0..n).map(|_| T::from_f64(rng.range(-1.0, 1.0))).collect();
        let mut y = vec![T::zero(); n];
        if i < 8 {
            tables.m2m(i, &x, &mut y, &mut scratch);
            record(&y);
            y.fill(T::zero());
            tables.l2l(i, &x, &mut y, &mut scratch);
            record(&y);
            y.fill(T::zero());
        }
        tables.m2l(i, &x, &mut y, &mut scratch);
        record(&y);
    }
    bits
}

fn check_round_trip<T: Stored + core::fmt::Debug>(p: usize, strategy: M2lStrategy) {
    let cache = empty_cache(&format!("tables-{strategy:?}-p{p}-{}", T::PRECISION.name()));
    let built = Tables::<T>::build(p, strategy);
    let (first, outcomes) = Tables::<T>::load_or_build(p, strategy, &cache);
    let kinds: Vec<_> = outcomes.iter().map(|(kind, _)| *kind).collect();
    assert_eq!(kinds, built.kinds(), "{strategy:?}");
    for (kind, outcome) in &outcomes {
        assert!(
            matches!(outcome, CacheOutcome::Built),
            "{strategy:?} p = {p}: {kind:?} {outcome:?}"
        );
    }
    let (second, outcomes) = Tables::<T>::load_or_build(p, strategy, &cache);
    for (kind, outcome) in &outcomes {
        assert!(
            matches!(outcome, CacheOutcome::Loaded),
            "{strategy:?} p = {p}: {kind:?} {outcome:?}"
        );
    }
    assert_eq!(first, built);
    assert_eq!(second, built);
    let want = outputs(&built, 0x7841);
    assert_eq!(outputs(&first, 0x7841), want);
    assert_eq!(outputs(&second, 0x7841), want);
    eprintln!(
        "{strategy:?} p = {p} {}: built, then loaded; {} outputs bit-identical",
        T::PRECISION.name(),
        want.len()
    );
}

#[test]
fn load_or_build_reports_built_then_loaded_with_bit_identical_results() {
    // Error measure: exact equality of the tables and of the bits of every
    // translation's output, against `Tables::build` without a cache.
    for strategy in STRATEGIES {
        check_round_trip::<f64>(4, strategy);
    }
    check_round_trip::<f32>(3, M2lStrategy::Classes);
    check_round_trip::<f64>(0, M2lStrategy::Auto);
}

#[test]
#[ignore = "timings: run with `cargo test -p nd-fmm-exec --release -- --ignored`"]
fn table_build_and_load_times_at_p_8_and_16() {
    // Reports, per strategy, the time to build the tables without a cache, to build and
    // store them into an empty cache (cold), and to load them back (warm), in f64.
    for p in [8, 16] {
        for strategy in STRATEGIES {
            let start = Instant::now();
            let built = Tables::<f64>::build(p, strategy);
            let build = start.elapsed();
            let cache = empty_cache(&format!("timings-{strategy:?}-p{p}"));
            let start = Instant::now();
            let (cold, _) = Tables::<f64>::load_or_build(p, strategy, &cache);
            let cold_time = start.elapsed();
            let start = Instant::now();
            let (warm, outcomes) = Tables::<f64>::load_or_build(p, strategy, &cache);
            let warm_time = start.elapsed();
            assert!(
                outcomes
                    .iter()
                    .all(|(_, o)| matches!(o, CacheOutcome::Loaded))
            );
            assert!(cold == built && warm == built);
            eprintln!(
                "{strategy:?} p = {p}: build {build:.3?}, cold {cold_time:.3?}, warm load \
                 {warm_time:.3?}"
            );
        }
    }
}

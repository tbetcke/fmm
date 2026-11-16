//! The operator's tables built once (Helmholtz CH4.0h): an operator from
//! `HelmholtzOperator::unbuilt` with the settings, then a pool and `build_tables`, has the
//! tables of `with_block_budget`'s operator with the same settings applied after it (the
//! chain of setters that rebuilds its tables, on a pool of the same threads, so that the
//! scratch sets compare), level for level and bit for bit through the
//! pair bodies; its RCR tables store M2L blocks only on the levels whose M2L is RCR's, and
//! the unbuilt operator reports the degrees given and no table.

use std::sync::Arc;

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{ChargeKind, Densities, HelmholtzM2l, HelmholtzOperator, Wavenumber};
use nd_fmm_math::Layout;
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::DEFAULT_BLOCK_BUDGET;
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

/// A source and a target on `level` whose offset is the V-list offset d.
fn pair(level: usize, d: usize) -> (MortonKey, MortonKey) {
    let offset = m2l_offsets()[d];
    let target = offset.map(|x| x.max(0) as usize);
    let source: [usize; 3] = core::array::from_fn(|a| (target[a] as i64 - offset[a]) as usize);
    let key = |i: [usize; 3]| morton::from_index_and_level(i, level);
    (key(source), key(target))
}

/// The dense budget of the cases: Batched's, Dense's and Classes' tables on the two finest
/// levels (p = 6 and 4), so that the debug run stays short.
const DENSE_BUDGET: usize = 64 << 20;

/// The settings of a case, applied in the builder's order.
fn settings(
    op: HelmholtzOperator<f64>,
    m2l: HelmholtzM2l,
    tolerance: Option<f64>,
) -> HelmholtzOperator<f64> {
    op.with_m2l_max_degree(Some(10))
        .with_m2l_plane_wave_min_degree(Some(11))
        .with_plane_wave_tolerance(tolerance)
        .with_m2l(m2l, DENSE_BUDGET)
}

/// Every M2L pair body on every level, every seventh offset, on random multipoles; then
/// every M2M and L2L pair body of one child per level.
fn pair_outputs(op: &mut HelmholtzOperator<f64>) -> Vec<f64> {
    let mut rng = Rng(11);
    let mut out = Vec::new();
    for level in 2..op.nlevels() {
        let multipole: Vec<f64> = (0..op.multipole_size(level)).map(|_| rng.next()).collect();
        for d in (0..M2L_OFFSET_COUNT).step_by(7) {
            let (source, target) = pair(level, d);
            let mut local = vec![0.0; op.local_size(level)];
            op.m2l(source, target, d, &multipole, &mut local);
            out.extend(local);
        }
    }
    for level in 1..op.nlevels() {
        let child = morton::from_index_and_level([1, 0, 1], level);
        let parent = morton::parent(child).expect("level ≥ 1");
        let multipole: Vec<f64> = (0..op.multipole_size(level)).map(|_| rng.next()).collect();
        let mut up = vec![0.0; op.multipole_size(level - 1)];
        op.m2m(
            child,
            parent,
            morton::child_index(child),
            &multipole,
            &mut up,
        );
        out.extend(up);
        let local: Vec<f64> = (0..op.local_size(level - 1)).map(|_| rng.next()).collect();
        let mut down = vec![0.0; op.local_size(level)];
        op.l2l(parent, child, morton::child_index(child), &local, &mut down);
        out.extend(down);
    }
    out
}

#[test]
fn the_unbuilt_operator_builds_the_settings_tables() {
    // k̂₀ = 15; below 10⁻⁷ Auto raises its levels with waves to 32.
    let k = Wavenumber::new(30.0, 0.0).unwrap();
    let degrees = [0, 0, 14, 9, 6, 4];
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let cases = [
        (HelmholtzM2l::Rotation, None),
        (HelmholtzM2l::Auto, Some(1e-6)),
        (HelmholtzM2l::Auto, Some(5e-8)),
        (HelmholtzM2l::PlaneWave, Some(1e-6)),
        (HelmholtzM2l::Batched, None),
        (HelmholtzM2l::Dense, None),
        (HelmholtzM2l::Classes, None),
    ];
    for (m2l, tolerance) in cases {
        let what = format!("{m2l:?} at {tolerance:?}");
        let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
        let mut chain = settings(
            HelmholtzOperator::with_block_budget(
                k,
                densities,
                &domain(),
                &degrees,
                false,
                8,
                DEFAULT_BLOCK_BUDGET,
            )
            .unwrap()
            .with_pool(Arc::clone(&pool)),
            m2l,
            tolerance,
        );
        let unbuilt = settings(
            HelmholtzOperator::unbuilt(
                k,
                densities,
                &domain(),
                &degrees,
                false,
                8,
                DEFAULT_BLOCK_BUDGET,
            )
            .unwrap(),
            m2l,
            tolerance,
        );
        // Unbuilt: the degrees given, no M2L table, nothing timed.
        for (l, &p) in degrees.iter().enumerate() {
            assert_eq!(unbuilt.degree(l), p, "{what}");
            assert!(!unbuilt.tables().has_m2l(l), "{what}");
        }
        assert!(!unbuilt.tables().has_translations(), "{what}");
        assert_eq!(unbuilt.table_timings().rcr.as_nanos(), 0, "{what}");
        let mut once = unbuilt.with_pool(Arc::clone(&pool)).build_tables();
        assert_eq!(once.nlevels(), chain.nlevels());
        for l in 0..degrees.len() {
            assert_eq!(once.degree(l), chain.degree(l), "{what}, level {l}");
            assert_eq!(
                once.translation_degree(l),
                chain.translation_degree(l),
                "{what}, level {l}"
            );
            assert_eq!(once.m2l_on(l), chain.m2l_on(l), "{what}, level {l}");
            let stored = once.tables().m2l_blocks_stored(l);
            assert_eq!(
                stored,
                chain.tables().m2l_blocks_stored(l),
                "{what}, level {l}"
            );
            // Blocks only where the M2L is RCR's, and there within the budget (these fit).
            assert_eq!(
                stored,
                l >= 2 && once.m2l_on(l) == HelmholtzM2l::Rotation,
                "{what}, level {l}"
            );
        }
        // Each M2L of its own runs on some level (so its tables were built).
        if m2l != HelmholtzM2l::Auto {
            assert!((2..degrees.len()).any(|l| once.m2l_on(l) == m2l), "{what}");
        }
        // And RCR on the levels Batched's and Dense's budget leaves out (Auto and PlaneWave
        // may give every level waves or Batched, and the class tables fit every level).
        if matches!(m2l, HelmholtzM2l::Batched | HelmholtzM2l::Dense) {
            assert!(
                (2..degrees.len()).any(|l| once.m2l_on(l) == HelmholtzM2l::Rotation),
                "{what}: some level on RCR"
            );
        }
        assert_eq!(once.tables().bytes(), chain.tables().bytes(), "{what}");
        assert_eq!(once.dense_bytes(), chain.dense_bytes(), "{what}");
        assert_eq!(once.class_bytes(), chain.class_bytes(), "{what}");
        assert_eq!(once.batched_bytes(), chain.batched_bytes(), "{what}");
        assert_eq!(once.plane_wave_bytes(), chain.plane_wave_bytes(), "{what}");
        let (a, b) = (pair_outputs(&mut once), pair_outputs(&mut chain));
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x == y, "{what}: value {i}: {x:e} against {y:e}");
        }
        let n = Layout::new(degrees[2]).len();
        assert!(a.len() > n, "{what}");
    }
}

/// An operator from `with_block_budget` is built: `build_tables` changes nothing.
#[test]
fn build_tables_on_a_built_operator_does_nothing() {
    let k = Wavenumber::new(4.0, 0.5).unwrap();
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let op =
        HelmholtzOperator::<f64>::new(k, densities, &domain(), &[0, 0, 6, 5], false, 8).unwrap();
    let before = op.table_timings();
    let op = op.build_tables();
    assert_eq!(op.table_timings(), before);
    assert!(op.tables().has_translations());
}

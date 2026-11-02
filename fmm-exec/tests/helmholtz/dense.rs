//! The operator's Dense M2L (Helmholtz CH3.2a): pair by pair against Rotation on every
//! offset of levels 2–4, real and complex k, real and complex charges, two vectors, relative
//! L2 over the local within [`DENSE_BOUND`]; the levels chosen by the budget; a budget of
//! zero is Rotation bit for bit. The FMM under Dense is tested in `tests/helmholtz_fmm.rs`.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    ChargeKind, DEFAULT_DENSE_BUDGET, Densities, HelmholtzM2l, HelmholtzOperator, Wavenumber,
};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

const LEVELS: usize = 5;

/// Dense against Rotation, relative L2 over a local: the tables' agreement (4e-14 in the
/// FMM-weighted measure in `nd-fmm-tables`) with room for the plain L2 of random inputs.
const DENSE_BOUND: f64 = 1e-12;

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

/// Named degrees, small enough for the dense tables in a debug build (the accuracy of the
/// pairs is not the point: Dense against Rotation at the same degrees is).
fn degrees(_k: &Wavenumber) -> Vec<usize> {
    vec![0, 0, 6, 8, 5]
}

fn operator(
    k: Wavenumber,
    kind: ChargeKind,
    m: usize,
    m2l: HelmholtzM2l,
    budget: usize,
) -> HelmholtzOperator<f64> {
    let densities = Densities::new(k.kind(), kind, m);
    HelmholtzOperator::new(k, densities, &domain(), &degrees(&k), false, 8)
        .unwrap()
        .with_m2l(m2l, budget)
}

/// A source and a target on `level` whose offset is the V-list offset d.
fn pair(level: usize, d: usize) -> (MortonKey, MortonKey) {
    let offset = m2l_offsets()[d];
    let target = offset.map(|x| x.max(0) as usize);
    let source: [usize; 3] = core::array::from_fn(|a| (target[a] as i64 - offset[a]) as usize);
    let key = |i: [usize; 3]| morton::from_index_and_level(i, level);
    (key(source), key(target))
}

/// Every offset of levels 2–4 through `a` and `b` from the same multipoles: the largest
/// relative L2 of a's local against b's, and whether every local was the same bit for bit.
fn compare(a: &mut HelmholtzOperator<f64>, b: &mut HelmholtzOperator<f64>) -> (f64, bool) {
    let mut rng = Rng(5);
    let (mut worst, mut same) = (0.0f64, true);
    for level in 2..LEVELS {
        let multipole: Vec<f64> = (0..a.multipole_size(level)).map(|_| rng.next()).collect();
        for d in 0..M2L_OFFSET_COUNT {
            let (source, target) = pair(level, d);
            let start: Vec<f64> = (0..a.local_size(level)).map(|_| rng.next()).collect();
            let (mut x, mut y) = (start.clone(), start.clone());
            a.m2l(source, target, d, &multipole, &mut x);
            b.m2l(source, target, d, &multipole, &mut y);
            same &= x == y;
            let diff: f64 = x
                .iter()
                .zip(&y)
                .zip(&start)
                .map(|((u, v), s)| ((u - s) - (v - s)).powi(2))
                .sum();
            let norm: f64 = y.iter().zip(&start).map(|(v, s)| (v - s).powi(2)).sum();
            worst = worst.max((diff / norm).sqrt());
        }
    }
    (worst, same)
}

#[test]
fn dense_is_rotation_to_rounding() {
    // One and two multipole planes for real k; complex k has two planes for either kind.
    let cases = [
        (20.0, 0.0, ChargeKind::Real),
        (20.0, 0.0, ChargeKind::Complex),
        (20.0, 2.0, ChargeKind::Real),
        (0.5, 0.0, ChargeKind::Real),
    ];
    for (re, im, kind) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let mut dense = operator(k, kind, 2, HelmholtzM2l::Dense, usize::MAX);
        let mut rotation = operator(k, kind, 2, HelmholtzM2l::Rotation, usize::MAX);
        assert_eq!(dense.m2l_choice(), HelmholtzM2l::Dense);
        for level in 2..LEVELS {
            assert_eq!(dense.m2l_on(level), HelmholtzM2l::Dense);
            assert_eq!(rotation.m2l_on(level), HelmholtzM2l::Rotation);
        }
        let (worst, _) = compare(&mut dense, &mut rotation);
        println!("k = {re} + {im}i, {kind:?}: Dense against Rotation {worst:.2e}");
        assert!(
            worst <= DENSE_BOUND,
            "k = {re} + {im}i, {kind:?}: {worst:e}"
        );
    }
}

#[test]
fn the_budget_chooses_the_levels() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let degrees = degrees(&k);
    // The finest level and the coarsest fit, not the middle one: a budget of levels 4 and 2.
    let budget =
        DenseM2l::<f64>::level_bytes(degrees[4]) + DenseM2l::<f64>::level_bytes(degrees[2]);
    assert!(
        degrees[3] >= degrees[2],
        "the middle level's tables are the larger"
    );
    let op = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Dense, budget);
    let chosen = DenseM2l::<f64>::select(&degrees, budget);
    for (level, &on) in chosen.iter().enumerate() {
        let want = if on {
            HelmholtzM2l::Dense
        } else {
            HelmholtzM2l::Rotation
        };
        assert_eq!(op.m2l_on(level), want, "level {level}");
    }
    assert_eq!(chosen, [false, false, true, false, true]);
    assert_eq!(op.dense_bytes(), budget);
    let rotation = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Rotation, budget);
    assert_eq!(rotation.dense_bytes(), 0);
    assert_eq!(rotation.m2l_choice(), HelmholtzM2l::Rotation);
    assert!(DEFAULT_DENSE_BUDGET >= DenseM2l::<f64>::level_bytes(15));
}

/// Under Dense with a budget of zero no level has tables: Rotation bit for bit.
#[test]
fn no_budget_is_rotation_bit_for_bit() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let mut dense = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Dense, 0);
    let mut rotation = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Rotation, 0);
    assert_eq!(dense.dense_bytes(), 0);
    let (_, same) = compare(&mut dense, &mut rotation);
    assert!(same);
}

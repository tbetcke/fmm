//! The operator's Dense M2L (Helmholtz CH3.2a), Classes M2L (Helmholtz CH3.2b) and Batched
//! M2L (Helmholtz CH3.2c): pair by pair against Rotation (and Classes and Batched against
//! Dense) on every offset of levels 2–4, real and complex k, real and complex charges, two
//! vectors, relative L2 over the local within [`DENSE_BOUND`] and [`CLASSES_BOUND`];
//! Batched bit for bit Dense with real multipoles (separate mode), fused within the bound,
//! and in both modes the same bits on every ISA; the
//! levels chosen by the budget; a budget of zero is Rotation bit for bit. The rule per level
//! (Helmholtz CH3.2d), level by level: fused Batched bit for bit on the levels of degree at
//! most the crossover, Rotation bit for bit above. The FMM under each is tested in
//! `tests/helmholtz_fmm.rs`.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    ChargeKind, DEFAULT_DENSE_BUDGET, Densities, HelmholtzM2l, HelmholtzOperator, Wavenumber,
    auto_batched_max_degree,
};
use nd_fmm_exec::operator::{Isa, MulAdd, P2pChoice};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::{ClassM2l, CubeTransforms};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

const LEVELS: usize = 5;

/// Dense against Rotation, relative L2 over a local: the same coefficients summed in
/// another order, so rounding (at most 7.2e-16 on the M6), with a margin of about ten.
const DENSE_BOUND: f64 = 1e-14;

/// Classes against Rotation and Dense, relative L2 over a local: two more transforms per
/// pair, so more rounding (at most 5.6e-15 on the M6), with a margin of about ten.
const CLASSES_BOUND: f64 = 5e-14;

/// Batched against Dense and Rotation with complex multipoles, and in the fused mode,
/// relative L2 over a local: the real plane's products before the imaginary plane's, or one
/// rounding per step, so rounding of Dense's own size (at most 7.4e-16 separate and 6.9e-16
/// fused against Dense on the M6), with a margin of about ten.
const BATCHED_BOUND: f64 = DENSE_BOUND;

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

/// Every offset of `level` through `a` and `b` from the same multipoles: whether every
/// local was the same bit for bit.
fn same_on(a: &mut HelmholtzOperator<f64>, b: &mut HelmholtzOperator<f64>, level: usize) -> bool {
    let mut rng = Rng(5 + level as u64);
    let multipole: Vec<f64> = (0..a.multipole_size(level)).map(|_| rng.next()).collect();
    (0..M2L_OFFSET_COUNT).all(|d| {
        let (source, target) = pair(level, d);
        let start: Vec<f64> = (0..a.local_size(level)).map(|_| rng.next()).collect();
        let (mut x, mut y) = (start.clone(), start);
        a.m2l(source, target, d, &multipole, &mut x);
        b.m2l(source, target, d, &multipole, &mut y);
        x == y
    })
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
        degrees[3] > degrees[2],
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

#[test]
fn classes_are_rotation_and_dense_to_rounding() {
    let cases = [
        (20.0, 0.0, ChargeKind::Real),
        (20.0, 0.0, ChargeKind::Complex),
        (20.0, 2.0, ChargeKind::Real),
        (0.5, 0.0, ChargeKind::Real),
    ];
    for (re, im, kind) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let mut classes = operator(k, kind, 2, HelmholtzM2l::Classes, usize::MAX);
        let mut dense = operator(k, kind, 2, HelmholtzM2l::Dense, usize::MAX);
        let mut rotation = operator(k, kind, 2, HelmholtzM2l::Rotation, usize::MAX);
        assert_eq!(classes.m2l_choice(), HelmholtzM2l::Classes);
        for level in 2..LEVELS {
            assert_eq!(classes.m2l_on(level), HelmholtzM2l::Classes);
        }
        assert_eq!(classes.dense_bytes(), 0);
        let (against_rotation, _) = compare(&mut classes, &mut rotation);
        let (against_dense, _) = compare(&mut classes, &mut dense);
        println!(
            "k = {re} + {im}i, {kind:?}: Classes against Rotation {against_rotation:.2e}, \
             against Dense {against_dense:.2e}"
        );
        assert!(against_rotation <= CLASSES_BOUND, "{against_rotation:e}");
        assert!(against_dense <= CLASSES_BOUND, "{against_dense:e}");
    }
}

#[test]
fn the_budget_chooses_the_class_levels() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let degrees = degrees(&k);
    let chosen = ClassM2l::<f64>::select(&degrees, usize::MAX);
    assert_eq!(chosen, [false, false, true, true, true]);
    let all = (2..LEVELS)
        .map(|l| ClassM2l::<f64>::level_bytes(degrees[l]))
        .sum::<usize>()
        + CubeTransforms::<f64>::bytes_for(*degrees[2..].iter().max().unwrap());
    // One byte short of everything drops a level.
    let budget = all - 1;
    let chosen = ClassM2l::<f64>::select(&degrees, budget);
    let op = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Classes, budget);
    for (level, &on) in chosen.iter().enumerate() {
        let want = if on {
            HelmholtzM2l::Classes
        } else {
            HelmholtzM2l::Rotation
        };
        assert_eq!(op.m2l_on(level), want, "level {level}");
    }
    assert_ne!(chosen, [false, false, true, true, true]);
    assert!(op.class_bytes() <= budget && op.class_bytes() > 0);
}

/// Under Classes with a budget of zero no level has tables: Rotation bit for bit.
#[test]
fn no_class_budget_is_rotation_bit_for_bit() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let mut classes = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Classes, 0);
    let mut rotation = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Rotation, 0);
    assert_eq!(classes.class_bytes(), 0);
    let (_, same) = compare(&mut classes, &mut rotation);
    assert!(same);
}

/// Batched pair by pair: with real multipoles (real k, real charges) Dense bit for bit, with
/// complex ones Dense and Rotation to rounding.
#[test]
fn batched_is_dense_bit_for_bit_with_real_multipoles() {
    let cases = [
        (20.0, 0.0, ChargeKind::Real, true),
        (0.5, 0.0, ChargeKind::Real, true),
        (20.0, 0.0, ChargeKind::Complex, false),
        (20.0, 2.0, ChargeKind::Real, false),
    ];
    for (re, im, kind, one_plane) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let mut batched = operator(k, kind, 2, HelmholtzM2l::Batched, usize::MAX);
        let mut dense = operator(k, kind, 2, HelmholtzM2l::Dense, usize::MAX);
        let mut rotation = operator(k, kind, 2, HelmholtzM2l::Rotation, usize::MAX);
        assert_eq!(batched.m2l_choice(), HelmholtzM2l::Batched);
        for level in 2..LEVELS {
            assert_eq!(batched.m2l_on(level), HelmholtzM2l::Batched);
        }
        assert_eq!(batched.batched_bytes(), dense.dense_bytes());
        assert_eq!(batched.dense_bytes(), 0);
        let (against_dense, same) = compare(&mut batched, &mut dense);
        let (against_rotation, _) = compare(&mut batched, &mut rotation);
        println!(
            "k = {re} + {im}i, {kind:?}: Batched against Dense {against_dense:.2e} (bit for \
             bit: {same}), against Rotation {against_rotation:.2e}"
        );
        assert_eq!(
            same, one_plane,
            "k = {re} + {im}i, {kind:?}: bit for bit Dense"
        );
        assert!(against_dense <= BATCHED_BOUND, "{against_dense:e}");
        assert!(against_rotation <= BATCHED_BOUND, "{against_rotation:e}");
    }
}

/// Batched in the fused mode pair by pair: Dense and Rotation to rounding with one and two
/// planes; the mode survives `with_p2p`.
#[test]
fn fused_batched_is_dense_to_rounding() {
    let cases = [
        (20.0, 0.0, ChargeKind::Real),
        (20.0, 2.0, ChargeKind::Complex),
    ];
    for (re, im, kind) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let mut fused = operator(k, kind, 2, HelmholtzM2l::Batched, usize::MAX)
            .with_m2l_mul_add(MulAdd::Fused)
            .with_p2p(P2pChoice::Reference)
            .unwrap();
        assert_eq!(fused.m2l_mul_add(), MulAdd::Fused);
        let mut dense = operator(k, kind, 2, HelmholtzM2l::Dense, usize::MAX);
        let mut rotation = operator(k, kind, 2, HelmholtzM2l::Rotation, usize::MAX);
        let (against_dense, same) = compare(&mut fused, &mut dense);
        let (against_rotation, _) = compare(&mut fused, &mut rotation);
        println!(
            "k = {re} + {im}i, {kind:?}: fused Batched against Dense {against_dense:.2e}, \
             against Rotation {against_rotation:.2e}"
        );
        assert!(
            !same,
            "the fused mode gave Dense's bits: is it engaged? (its steps round once)"
        );
        assert!(against_dense <= BATCHED_BOUND, "{against_dense:e}");
        assert!(against_rotation <= BATCHED_BOUND, "{against_rotation:e}");
    }
}

/// In both modes the column kernel gives the same bits on every ISA this machine runs, and
/// for the reference P2P (the scalar path). Pair by pair, so the kernel's groups of fewer
/// than four columns; its four-column groups are compared across ISAs in `nd-fmm-simd`'s
/// tests, and the level call against the pair body in `tests/helmholtz_fmm.rs`.
#[test]
fn batched_is_the_same_on_every_isa() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    for mode in [MulAdd::Separate, MulAdd::Fused] {
        let batched = || {
            operator(k, ChargeKind::Complex, 2, HelmholtzM2l::Batched, usize::MAX)
                .with_m2l_mul_add(mode)
        };
        let mut scalar = batched().with_p2p(P2pChoice::Isa(Isa::Scalar)).unwrap();
        let choices = Isa::all()
            .filter(|isa| isa.is_available())
            .map(P2pChoice::Isa)
            .chain([P2pChoice::Reference, P2pChoice::Auto]);
        for choice in choices {
            let mut other = batched().with_p2p(choice).unwrap();
            assert_eq!(other.m2l_mul_add(), mode);
            let (_, same) = compare(&mut other, &mut scalar);
            assert!(same, "{mode}, {choice}: not the scalar path's bits");
        }
    }
}

#[test]
fn the_budget_chooses_the_batched_levels() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let degrees = degrees(&k);
    let budget =
        DenseM2l::<f64>::level_bytes(degrees[4]) + DenseM2l::<f64>::level_bytes(degrees[2]);
    let op = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Batched, budget);
    let chosen = DenseM2l::<f64>::select(&degrees, budget);
    assert_eq!(chosen, [false, false, true, false, true]);
    for (level, &on) in chosen.iter().enumerate() {
        let want = if on {
            HelmholtzM2l::Batched
        } else {
            HelmholtzM2l::Rotation
        };
        assert_eq!(op.m2l_on(level), want, "level {level}");
    }
    assert_eq!(op.batched_bytes(), budget);
}

/// Under Batched with a budget of zero no level has tables: Rotation bit for bit.
#[test]
fn no_batched_budget_is_rotation_bit_for_bit() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let mut batched = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Batched, 0);
    let mut rotation = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Rotation, 0);
    assert_eq!(batched.batched_bytes(), 0);
    let (_, same) = compare(&mut batched, &mut rotation);
    assert!(same);
}

/// The rule per level with a named crossover that splits the levels (degrees 6, 8, 5 on
/// levels 2–4, crossover 6): fused Batched bit for bit on levels 2 and 4, Rotation bit for
/// bit on level 3, with one and two planes and two vectors.
#[test]
fn auto_splits_the_levels_at_the_crossover() {
    let cases = [
        (20.0, 0.0, ChargeKind::Real),
        (20.0, 0.0, ChargeKind::Complex),
        (20.0, 2.0, ChargeKind::Real),
    ];
    for (re, im, kind) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let degrees = degrees(&k);
        let mut auto = operator(k, kind, 2, HelmholtzM2l::Auto, usize::MAX)
            .with_m2l_max_degree(Some(6))
            .with_p2p(P2pChoice::Reference)
            .unwrap();
        let mut fused = operator(k, kind, 2, HelmholtzM2l::Batched, usize::MAX)
            .with_m2l_mul_add(MulAdd::Fused)
            .with_p2p(P2pChoice::Reference)
            .unwrap();
        let mut rotation = operator(k, kind, 2, HelmholtzM2l::Rotation, usize::MAX);
        assert_eq!(auto.m2l_choice(), HelmholtzM2l::Auto);
        assert_eq!(auto.max_degree(), Some(6));
        assert_eq!(auto.m2l_mul_add(), MulAdd::Fused);
        assert_eq!(
            auto.batched_bytes(),
            BatchedM2l::<f64>::level_bytes(degrees[2]) + BatchedM2l::<f64>::level_bytes(degrees[4])
        );
        for (level, &p) in degrees.iter().enumerate().take(LEVELS).skip(2) {
            let batched = p <= 6;
            let want = if batched {
                HelmholtzM2l::Batched
            } else {
                HelmholtzM2l::Rotation
            };
            assert_eq!(auto.m2l_on(level), want, "k = {re} + {im}i, level {level}");
            let same = if batched {
                same_on(&mut auto, &mut fused, level)
            } else {
                same_on(&mut auto, &mut rotation, level)
            };
            assert!(
                same,
                "k = {re} + {im}i, {kind:?}, level {level}: not {want:?}'s bits"
            );
        }
    }
}

/// A crossover above every degree is fused Batched bit for bit; below every degree,
/// Rotation bit for bit with no tables; a named mode is kept; the other settings have no
/// crossover.
#[test]
fn auto_at_the_extremes_and_its_mode() {
    let k = Wavenumber::new(20.0, 2.0).unwrap();
    let auto = |max_degree| {
        operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Auto, usize::MAX)
            .with_m2l_max_degree(Some(max_degree))
    };
    let mut every = auto(usize::MAX);
    let mut fused = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Batched, usize::MAX)
        .with_m2l_mul_add(MulAdd::Fused);
    assert!((2..LEVELS).all(|l| every.m2l_on(l) == HelmholtzM2l::Batched));
    assert!(compare(&mut every, &mut fused).1);

    let mut none = auto(4);
    let mut rotation = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Rotation, 0);
    assert_eq!(none.batched_bytes(), 0);
    assert!((2..LEVELS).all(|l| none.m2l_on(l) == HelmholtzM2l::Rotation));
    assert!(compare(&mut none, &mut rotation).1);

    let mut separate = auto(6).with_m2l_mul_add(MulAdd::Separate);
    let mut batched = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Batched, usize::MAX);
    assert_eq!(separate.m2l_mul_add(), MulAdd::Separate);
    assert!(same_on(&mut separate, &mut batched, 2));

    // A mode named before `with_m2l` holds; unnamed, each setting takes its own default.
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let named = HelmholtzOperator::<f64>::new(k, densities, &domain(), &degrees(&k), false, 8)
        .unwrap()
        .with_m2l_mul_add(MulAdd::Separate)
        .with_m2l(HelmholtzM2l::Auto, 0);
    assert_eq!(named.m2l_mul_add(), MulAdd::Separate);
    let back = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Auto, 0)
        .with_m2l(HelmholtzM2l::Batched, 0);
    assert_eq!(back.m2l_mul_add(), MulAdd::Separate);

    // A budget binds under Auto as under Batched: the finest admitted level first.
    let budget = BatchedM2l::<f64>::level_bytes(5);
    let tight = operator(k, ChargeKind::Complex, 1, HelmholtzM2l::Auto, budget)
        .with_m2l_max_degree(Some(6));
    assert_eq!(tight.m2l_on(4), HelmholtzM2l::Batched);
    assert_eq!(tight.m2l_on(2), HelmholtzM2l::Rotation);

    for m2l in [
        HelmholtzM2l::Rotation,
        HelmholtzM2l::Dense,
        HelmholtzM2l::Classes,
        HelmholtzM2l::Batched,
    ] {
        assert_eq!(
            operator(k, ChargeKind::Complex, 1, m2l, 0).max_degree(),
            None
        );
    }
}

/// Without a named crossover the rule's, of the column kernel's ISA and the planes; set
/// again by `with_p2p`, so either order of the calls gives the same levels and bits.
#[test]
fn auto_takes_the_rule_of_the_kernel() {
    let cases = [
        (20.0, 0.0, ChargeKind::Real, 1),
        (20.0, 2.0, ChargeKind::Complex, 2),
    ];
    for (re, im, kind, planes) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        let degrees = degrees(&k);
        let detected = operator(k, kind, 1, HelmholtzM2l::Auto, usize::MAX);
        assert_eq!(
            detected.max_degree(),
            Some(auto_batched_max_degree(Isa::detect(), planes))
        );
        let choices = Isa::all()
            .filter(|isa| isa.is_available())
            .map(P2pChoice::Isa)
            .chain([P2pChoice::Reference]);
        for choice in choices {
            let isa = match choice {
                P2pChoice::Isa(isa) => isa,
                _ => Isa::Scalar,
            };
            let rule = auto_batched_max_degree(isa, planes);
            let mut after = operator(k, kind, 1, HelmholtzM2l::Auto, usize::MAX)
                .with_p2p(choice)
                .unwrap();
            let densities = Densities::new(k.kind(), kind, 1);
            let mut before = HelmholtzOperator::new(k, densities, &domain(), &degrees, false, 8)
                .unwrap()
                .with_p2p(choice)
                .unwrap()
                .with_m2l(HelmholtzM2l::Auto, usize::MAX);
            assert_eq!(after.max_degree(), Some(rule), "{choice}");
            assert_eq!(before.max_degree(), Some(rule), "{choice}");
            let chosen = BatchedM2l::<f64>::select_within(&degrees, usize::MAX, rule);
            for (level, &on) in chosen.iter().enumerate().skip(2) {
                let want = if on {
                    HelmholtzM2l::Batched
                } else {
                    HelmholtzM2l::Rotation
                };
                assert_eq!(after.m2l_on(level), want, "{choice}, level {level}");
                assert_eq!(before.m2l_on(level), want, "{choice}, level {level}");
            }
            assert!(compare(&mut after, &mut before).1, "{choice}");
        }
    }
}

/// The crossovers: one plane at least two planes' on every ISA (a real multipole costs
/// half the products), AVX-512's at least NEON's and AVX2's, the scalar kernel's the
/// smallest.
#[test]
fn the_crossovers_are_ordered() {
    for isa in Isa::all() {
        assert!(
            auto_batched_max_degree(isa, 1) >= auto_batched_max_degree(isa, 2),
            "{isa}"
        );
        for planes in [1, 2] {
            assert!(
                auto_batched_max_degree(Isa::Scalar, planes)
                    <= auto_batched_max_degree(isa, planes)
            );
            assert!(
                auto_batched_max_degree(Isa::Avx512, planes)
                    >= auto_batched_max_degree(isa, planes)
            );
        }
    }
}

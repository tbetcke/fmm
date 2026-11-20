//! The operator's diagonal M2L (Helmholtz CH4.2b1): the rule per level (the classes with
//! E(l, c) ≤ θ ε by the constants of the transfer functions' build (Helmholtz CH4.2c2), none
//! when off, without a tolerance, in f32 below 10⁻⁴, on levels whose M2L is not RCR's, or where
//! the scratch budget binds), and the pair body against Rotation's within 2E in the measure of
//! CH1.3d1 (P2M of charges in the source box, L2P at points of the target box, relative ℓ2 of
//! the potential), with the classes on RCR Rotation's bits, by both builds. The level call in
//! the FMM is tested in `tests/helmholtz_fmm.rs` (`check_diagonal`).

use nd_fmm_exec::geometry::{self, Domain, radius};
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, DEFAULT_DIAGONAL_BUILD, DIAGONAL_F32_MIN_TOLERANCE, Densities,
    DiagonalCosts, DiagonalShape, HelmholtzDiagonal, HelmholtzM2l, HelmholtzOperator,
    TransferBuild, Wavenumber, degree, diagonal_auto_in_range, diagonal_beats_rcr,
    diagonal_chunk_targets, diagonal_rule_classes, s7_degree,
};
use nd_fmm_math::Complex;
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_ref::helmholtz::diagonal::{BoundConstants, THETA, class_of, classes};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::classes::{ClassM2l as Classes, CubeTransforms};
use nd_fmm_tables::helmholtz::diagonal::{BLOCK_BOXES, class_bounds};
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

const LEVELS: usize = 5;

/// The tolerance of the tests' rule: at k = 40 on the unit cube level 2 (k a = 8.7) takes the
/// far classes and leaves the nearest on RCR.
const EPSILON: f64 = 1e-4;

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

/// S7's per-pair degrees at ε (the tests are single pairs).
fn degrees(k: &Wavenumber, epsilon: f64) -> Vec<usize> {
    (0..LEVELS)
        .map(|l| match l {
            0 | 1 => 0,
            _ => s7_degree(k, 3f64.sqrt() * radius(l, &domain()), epsilon),
        })
        .collect()
}

fn operator<T: nd_fmm_exec::operator::SimdScalar>(
    k: Wavenumber,
    kind: ChargeKind,
    m: usize,
    diagonal: HelmholtzDiagonal,
) -> HelmholtzOperator<T> {
    let densities = Densities::new(k.kind(), kind, m);
    HelmholtzOperator::<T>::new(k, densities, &domain(), &degrees(&k, EPSILON), false, 64)
        .unwrap()
        .with_m2l(HelmholtzM2l::Rotation, 0)
        .with_diagonal(diagonal)
}

/// The level's wavenumber in its units, k r_l.
fn k_hat(k: &Wavenumber, level: usize) -> Complex<f64> {
    let r = radius(level, &domain());
    Complex::new(k.re() * r, k.im() * r)
}

/// The class index (into `classes()`) of the V offset o.
fn class_index(o: [i64; 3]) -> usize {
    let (rep, _) = class_of(o.map(|x| x as i32));
    classes()
        .iter()
        .position(|c| c.representative == rep)
        .expect("every V offset has a class")
}

/// The rule's classes per level: those with E ≤ θ ε by the constants of the build, the same as
/// `diagonal_rule_classes`; the interpolated build's (C_T = 0.23) a subset of the direct build's
/// (0.15) on every level, and fewer on some (Helmholtz CH4.2c2).
#[test]
fn the_rule_takes_the_classes_within_the_bound() {
    let mut fewer = 0;
    for (re, im) in [(40.0, 0.0), (40.0, 4.0)] {
        let k = Wavenumber::new(re, im).unwrap();
        let p = degrees(&k, EPSILON);
        let mut by_build = Vec::new();
        for build in [TransferBuild::Direct, TransferBuild::Interpolated] {
            let op = operator::<f64>(
                k,
                ChargeKind::Complex,
                1,
                HelmholtzDiagonal::Tolerance(EPSILON),
            )
            .with_diagonal_build(build);
            assert_eq!(op.diagonal_build(), build);
            let mut some_level = false;
            let mut levels = Vec::new();
            for (l, &p_l) in p.iter().enumerate() {
                let expected: Vec<usize> = if l < 2 {
                    Vec::new()
                } else {
                    class_bounds(p_l, k_hat(&k, l), build.bound_constants(false))
                        .iter()
                        .enumerate()
                        .filter(|&(_, &e)| e <= THETA * EPSILON)
                        .map(|(c, _)| c)
                        .collect()
                };
                if l >= 2 {
                    assert_eq!(
                        diagonal_rule_classes::<f64>(p_l, k_hat(&k, l), EPSILON, build),
                        expected
                    );
                }
                let got = op.diagonal_classes(l).unwrap_or_default();
                println!("k = {re} + {im}i, {build:?}, level {l}, p {p_l}: classes {got:?}");
                assert_eq!(got, expected, "{build:?}, level {l}");
                some_level |= !got.is_empty();
                // The M2L reported stays Rotation's (the RCR pairs of the level).
                assert_eq!(op.m2l_on(l), HelmholtzM2l::Rotation);
                levels.push(got);
            }
            assert!(some_level, "{build:?}: some level takes the diagonal form");
            by_build.push(levels);
        }
        for (l, (direct, interpolated)) in by_build[0].iter().zip(&by_build[1]).enumerate() {
            assert!(
                interpolated.iter().all(|c| direct.contains(c)),
                "k = {re} + {im}i, level {l}: {interpolated:?} not within {direct:?}"
            );
            fewer += usize::from(interpolated.len() < direct.len());
        }
    }
    assert!(
        fewer > 0,
        "the interpolated build's rule took fewer classes on no level"
    );
    // Level 3 at k = 40, 10⁻⁴ (k a = 4.3): the near classes on RCR, so the level is mixed.
    let k = Wavenumber::new(40.0, 0.0).unwrap();
    let op = operator::<f64>(
        k,
        ChargeKind::Complex,
        1,
        HelmholtzDiagonal::Tolerance(EPSILON),
    );
    let level3 = op.diagonal_classes(3).expect("level 3 is diagonal");
    assert!(!level3.contains(&0) && level3.len() < 34, "{level3:?}");
}

#[test]
fn no_level_where_the_rule_does_not_apply() {
    let k = Wavenumber::new(40.0, 0.0).unwrap();
    let none = |op: &HelmholtzOperator<f64>| (0..LEVELS).all(|l| op.diagonal_classes(l).is_none());
    // Off (the default) and the rule without a tolerance.
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let default =
        HelmholtzOperator::<f64>::new(k, densities, &domain(), &degrees(&k, EPSILON), false, 64)
            .unwrap();
    assert_eq!(default.diagonal(), HelmholtzDiagonal::Off);
    // The default build (Helmholtz CH4.2c2's sign-off).
    assert_eq!(default.diagonal_build(), DEFAULT_DIAGONAL_BUILD);
    assert_eq!(DEFAULT_DIAGONAL_BUILD, TransferBuild::Interpolated);
    assert!(none(&default));
    let rule = operator::<f64>(k, ChargeKind::Complex, 1, HelmholtzDiagonal::Rule);
    assert!(none(&rule));
    // The rule with the FMM's tolerance is the named tolerance's.
    let rule = rule.with_plane_wave_tolerance(Some(EPSILON));
    let named = operator::<f64>(
        k,
        ChargeKind::Complex,
        1,
        HelmholtzDiagonal::Tolerance(EPSILON),
    );
    for l in 0..LEVELS {
        assert_eq!(rule.diagonal_classes(l), named.diagonal_classes(l));
    }
    // Levels with class tables are not RCR's (Classes: 16 matrices a level, Dense's check
    // at a fraction of its build in debug); a level without them is.
    let p4 = degrees(&k, EPSILON)[4];
    let one_level = Classes::<f64>::level_bytes(p4) + CubeTransforms::<f64>::bytes_for(p4);
    let classes = named.with_m2l(HelmholtzM2l::Classes, one_level);
    assert_eq!(classes.m2l_on(4), HelmholtzM2l::Classes);
    assert!(classes.diagonal_classes(4).is_none());
    assert_eq!(classes.m2l_on(3), HelmholtzM2l::Rotation);
    assert!(classes.diagonal_classes(3).is_some());
    // f32 below D3's 10⁻⁴; at 10⁻³ the f32 bound's classes.
    let below = operator::<f32>(
        k,
        ChargeKind::Complex,
        1,
        HelmholtzDiagonal::Tolerance(DIAGONAL_F32_MIN_TOLERANCE / 2.0),
    );
    assert!((0..LEVELS).all(|l| below.diagonal_classes(l).is_none()));
    for build in [TransferBuild::Direct, TransferBuild::Interpolated] {
        assert!(diagonal_rule_classes::<f32>(20, k_hat(&k, 2), 5e-5, build).is_empty());
    }
    let f32_op = operator::<f32>(
        k,
        ChargeKind::Complex,
        1,
        HelmholtzDiagonal::Tolerance(1e-3),
    );
    let p = degrees(&k, EPSILON);
    for (l, &p_l) in p.iter().enumerate().skip(2) {
        let expected: Vec<usize> = class_bounds(p_l, k_hat(&k, l), BoundConstants::F32)
            .iter()
            .enumerate()
            .filter(|&(_, &e)| e <= THETA * 1e-3)
            .map(|(c, _)| c)
            .collect();
        assert_eq!(f32_op.diagonal_classes(l).unwrap_or_default(), expected);
        // f32 tables are the f64 tables rounded: F32's constants by either build.
        assert_eq!(
            diagonal_rule_classes::<f32>(p_l, k_hat(&k, l), 1e-3, TransferBuild::Interpolated),
            expected
        );
    }
}

/// Costs under which the diagonal form beats every other M2L wherever a class passes (the
/// waves cost more per box than any level's RCR pairs).
const FREE: DiagonalCosts = DiagonalCosts {
    s2f: [0.0, 0.0],
    f2r: [0.0, 0.0],
    pair: [0.0, 0.0],
    rotation: [1.0, 1.0],
    plane_wave: [1e12, 1e12],
};

/// Costs under which it never does.
const DEAR: DiagonalCosts = DiagonalCosts {
    pair: [1e9, 1e9],
    plane_wave: [0.0, 0.0],
    ..FREE
};

/// The time rule (Helmholtz CH4.3b1) on the RCR levels: `HelmholtzDiagonal::Auto` at the FMM's
/// tolerance takes a level of `Rule`'s with its classes where `diagonal_beats_rcr` of the
/// costs and the level's shape (the uniform level's without shapes), and none elsewhere; the
/// budget's fallback first; none in f32 below 10⁻⁴.
#[test]
fn the_time_rule_on_the_rcr_levels() {
    let k = Wavenumber::new(40.0, 0.0).unwrap();
    let p = degrees(&k, EPSILON);
    let with = |costs: Option<DiagonalCosts>, shapes: Option<Vec<DiagonalShape>>| {
        operator::<f64>(k, ChargeKind::Complex, 1, HelmholtzDiagonal::Auto)
            .with_plane_wave_tolerance(Some(EPSILON))
            .with_diagonal_costs(costs)
            .with_diagonal_shapes(shapes)
    };
    let rule = operator::<f64>(k, ChargeKind::Complex, 1, HelmholtzDiagonal::Rule)
        .with_plane_wave_tolerance(Some(EPSILON));
    // Without a tolerance, none.
    let untold = operator::<f64>(k, ChargeKind::Complex, 1, HelmholtzDiagonal::Auto);
    assert!((0..LEVELS).all(|l| untold.diagonal_classes(l).is_none()));
    // The ISA's costs on the uniform level, by the model.
    let auto = with(None, None);
    assert_eq!(auto.diagonal(), HelmholtzDiagonal::Auto);
    assert!(auto.diagonal_shapes().is_none());
    let costs = auto.diagonal_costs();
    let mut some = false;
    for (l, &p_l) in p.iter().enumerate().skip(2) {
        let classes = rule.diagonal_classes(l).unwrap_or_default();
        let takes = diagonal_beats_rcr(&costs, p_l, &DiagonalShape::uniform(), &classes);
        println!("level {l}, p {p_l}: classes {classes:?}, by time {takes}");
        assert_eq!(
            auto.diagonal_classes(l),
            takes.then(|| classes.clone()),
            "level {l}"
        );
        some |= takes;
    }
    assert!(some, "the time rule takes some level of the cube at k = 40");
    // Named costs: free, every level of the rule's; dear, none.
    let free = with(Some(FREE), None);
    let dear = with(Some(DEAR), None);
    assert_eq!(free.diagonal_costs(), FREE);
    for l in 0..LEVELS {
        assert_eq!(
            free.diagonal_classes(l),
            rule.diagonal_classes(l),
            "level {l}"
        );
        assert!(dear.diagonal_classes(l).is_none(), "level {l}");
    }
    // The shapes: a level without V pairs takes nothing even at free costs; the others follow
    // their shapes.
    let mut shapes = vec![DiagonalShape::uniform(); LEVELS];
    shapes[3] = DiagonalShape::default();
    let shaped = with(Some(FREE), Some(shapes.clone()));
    assert_eq!(shaped.diagonal_shapes(), Some(&shapes[..]));
    assert!(shaped.diagonal_classes(3).is_none());
    assert_eq!(shaped.diagonal_classes(2), rule.diagonal_classes(2));
    // A level whose pairs all lie in classes the bound leaves on RCR: the conversions buy
    // nothing.
    let level3 = rule.diagonal_classes(3).expect("level 3 is diagonal");
    let failing = (0..34)
        .find(|c| !level3.contains(c))
        .expect("level 3 is mixed");
    let mut only_rcr = DiagonalShape {
        targets: 8,
        ..DiagonalShape::default()
    };
    only_rcr.class_pairs[failing] = 100;
    shapes[3] = only_rcr;
    let cheap = DiagonalCosts {
        s2f: [1.0, 0.0],
        ..FREE
    };
    assert!(
        with(Some(cheap), Some(shapes))
            .diagonal_classes(3)
            .is_none()
    );
    // The budget's fallback first.
    let tight = with(Some(FREE), None)
        .with_diagonal_sources(Some(vec![0, 0, 64, 512, 4096]))
        .with_diagonal_budget(Some(64 << 10));
    assert!(tight.diagonal_classes(2).is_none());
    // f32 below D3's 10⁻⁴.
    let below = operator::<f32>(k, ChargeKind::Complex, 1, HelmholtzDiagonal::Auto)
        .with_plane_wave_tolerance(Some(DIAGONAL_F32_MIN_TOLERANCE / 2.0))
        .with_diagonal_costs(Some(FREE));
    assert!((0..LEVELS).all(|l| below.diagonal_classes(l).is_none()));
}

/// Costs under which RCR beats every other M2L (the form's pair and the waves dear).
const RCR_FREE: DiagonalCosts = DiagonalCosts {
    pair: [1e9, 1e9],
    rotation: [0.0, 0.0],
    ..FREE
};

/// The time rule against the exponential M2L (Helmholtz CH4.3b1, CH4.3b2a): under Auto with
/// waves from every degree, `HelmholtzDiagonal::Auto` gives a level with waves in its range the
/// cheapest of the waves, the form and RCR by `wave_level_choice`, at the level's resolving
/// degree p_w, which it keeps whatever M2L runs (the defaults' degrees and translation degrees on
/// every level, so matched accuracy), with the bound's classes at p_w; where the form's scratch
/// does not fit the budget, the waves or RCR; below k a = 2.8 (`DIAGONAL_AUTO_MIN_KA`, where the
/// rule on an RCR level takes it) the waves. The FMM's calibrated degrees: at S7's per-pair
/// degrees no class passes at 10⁻⁷ (p = 40 at k a = 8.7).
#[test]
fn the_time_rule_against_the_waves() {
    let k = Wavenumber::new(40.0, 0.0).unwrap();
    let mut raised = false;
    let mut below = false;
    for epsilon in [EPSILON, 1e-7, 1e-8] {
        let p: Vec<usize> = (0..LEVELS)
            .map(|l| match l {
                0 | 1 => 0,
                _ => degree(&k, 3f64.sqrt() * radius(l, &domain()), epsilon),
            })
            .collect();
        let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
        let base = |diagonal: HelmholtzDiagonal, costs: DiagonalCosts| {
            HelmholtzOperator::<f64>::new(k, densities, &domain(), &p, false, 64)
                .unwrap()
                .with_plane_wave_tolerance(Some(epsilon))
                .with_m2l_plane_wave_min_degree(Some(0))
                .with_m2l(HelmholtzM2l::Auto, 0)
                .with_diagonal_costs(Some(costs))
                .with_diagonal(diagonal)
        };
        let off = base(HelmholtzDiagonal::Off, FREE);
        let rule = base(HelmholtzDiagonal::Rule, FREE);
        let free = base(HelmholtzDiagonal::Auto, FREE);
        let rcr = base(HelmholtzDiagonal::Auto, RCR_FREE);
        let dear = base(HelmholtzDiagonal::Auto, DEAR);
        // The scratch budget's fallback first: a level whose form would not fit runs the
        // cheaper of the waves and RCR (RCR at these costs), never the form.
        let tight = base(HelmholtzDiagonal::Auto, FREE)
            .with_diagonal_sources(Some(vec![0, 0, 64, 512, 4096]))
            .with_diagonal_budget(Some(64 << 10));
        let mut taken = 0;
        for (l, &p_l) in p.iter().enumerate().skip(2) {
            let waves = off.m2l_on(l) == HelmholtzM2l::PlaneWave;
            let p_w = off.degree(l);
            let classes =
                diagonal_rule_classes::<f64>(p_w, k_hat(&k, l), epsilon, DEFAULT_DIAGONAL_BUILD);
            println!(
                "ε {epsilon:e}, level {l}, p {p_l} (run {p_w}): waves {waves}, classes at p_w \
                 {classes:?}"
            );
            // Every setting keeps the defaults' degrees: the level's M2L alone changes.
            for (name, op) in [
                ("rule", &rule),
                ("free", &free),
                ("rcr", &rcr),
                ("dear", &dear),
                ("tight", &tight),
            ] {
                assert_eq!(op.degree(l), p_w, "{name}, level {l}");
                assert_eq!(
                    op.translation_degree(l),
                    off.translation_degree(l),
                    "{name}, level {l}"
                );
            }
            // Rule keeps the waves: the form only on RCR levels.
            assert_eq!(rule.m2l_on(l), off.m2l_on(l), "level {l}");
            // Dear costs (the waves free): Off's levels.
            assert_eq!(dear.m2l_on(l), off.m2l_on(l), "level {l}");
            assert!(dear.diagonal_classes(l).is_none() || !waves, "level {l}");
            let in_range = diagonal_auto_in_range(k_hat(&k, l));
            let classes_at_p =
                diagonal_rule_classes::<f64>(p_l, k_hat(&k, l), epsilon, DEFAULT_DIAGONAL_BUILD);
            below |= !in_range && !classes_at_p.is_empty();
            if !waves {
                // An RCR or Batched level: the time rule on the RCR levels
                // (`the_time_rule_on_the_rcr_levels`).
                assert_eq!(free.m2l_on(l), off.m2l_on(l), "level {l}");
                assert_eq!(rcr.m2l_on(l), off.m2l_on(l), "level {l}");
            } else if !in_range {
                // Below S6's range (k a < 2.8): the time rule leaves the waves, the rule on an
                // RCR level takes the form.
                for op in [&free, &rcr, &tight] {
                    assert_eq!(op.m2l_on(l), HelmholtzM2l::PlaneWave, "level {l}");
                    assert!(op.diagonal_classes(l).is_none(), "level {l}");
                }
                if !classes_at_p.is_empty() {
                    let rotation = |diagonal| {
                        HelmholtzOperator::<f64>::new(k, densities, &domain(), &p, false, 64)
                            .unwrap()
                            .with_plane_wave_tolerance(Some(epsilon))
                            .with_m2l(HelmholtzM2l::Rotation, 0)
                            .with_diagonal_costs(Some(FREE))
                            .with_diagonal(diagonal)
                    };
                    assert_eq!(
                        rotation(HelmholtzDiagonal::Rule).diagonal_classes(l),
                        Some(classes_at_p.clone()),
                        "level {l}"
                    );
                    assert!(
                        rotation(HelmholtzDiagonal::Auto)
                            .diagonal_classes(l)
                            .is_none(),
                        "level {l}"
                    );
                }
            } else {
                // Free form: the form with the bound's classes at p_w, or RCR where none passes.
                assert_eq!(free.m2l_on(l), HelmholtzM2l::Rotation, "level {l}");
                if classes.is_empty() {
                    assert!(free.diagonal_classes(l).is_none(), "level {l}");
                } else {
                    assert_eq!(free.diagonal_classes(l), Some(classes.clone()), "level {l}");
                    taken += 1;
                    raised |= p_w > p_l;
                }
                // Free RCR, and the tight budget at free form costs: RCR.
                for op in [&rcr, &tight] {
                    assert_eq!(op.m2l_on(l), HelmholtzM2l::Rotation, "level {l}");
                    assert!(op.diagonal_classes(l).is_none(), "level {l}");
                }
            }
        }
        assert!(
            taken > 0,
            "ε {epsilon:e}: some level with waves passes the bound"
        );
    }
    assert!(
        raised,
        "some level taken from the waves ran the form at the raised degree"
    );
    assert!(below, "some level below k a = 2.8 passes the bound");
}

/// The whole-level fallback: a level whose scratch at the agreed sources does not fit the
/// budget runs RCR; no sources (count 0) no tables.
#[test]
fn the_budget_falls_back_to_rcr_by_whole_levels() {
    let k = Wavenumber::new(40.0, 0.0).unwrap();
    let with = |budget: Option<usize>, sources: Option<Vec<usize>>| {
        operator::<f64>(
            k,
            ChargeKind::Complex,
            2,
            HelmholtzDiagonal::Tolerance(EPSILON),
        )
        .with_diagonal_sources(sources)
        .with_diagonal_budget(budget)
    };
    let unbounded = with(None, None);
    assert!(unbounded.diagonal_classes(2).is_some());
    // 64 sources at level 2: fits a generous budget, not a small one.
    let sources = vec![0, 0, 64, 512, 4096];
    let fits = with(Some(1 << 30), Some(sources.clone()));
    assert_eq!(fits.diagonal_classes(2), unbounded.diagonal_classes(2));
    let tight = with(Some(64 << 10), Some(sources));
    assert!(tight.diagonal_classes(2).is_none());
    // No V source on any rank: no tables.
    let empty = with(Some(1 << 30), Some(vec![0; LEVELS]));
    assert!((0..LEVELS).all(|l| empty.diagonal_classes(l).is_none()));
    assert_eq!(empty.diagonal_bytes(), 0);
    assert!(fits.diagonal_bytes() > 0);
}

#[test]
fn the_chunk_width() {
    assert_eq!(diagonal_chunk_targets(1000, 1), BLOCK_BOXES);
    assert_eq!(diagonal_chunk_targets(64, 72), 1);
    assert_eq!(diagonal_chunk_targets(64, 32), 2);
    assert_eq!(diagonal_chunk_targets(10_000, 8), BLOCK_BOXES);
}

/// A source and a target on `level` whose offset is the V-list offset d.
fn pair(level: usize, d: usize) -> (MortonKey, MortonKey) {
    let offset = m2l_offsets()[d];
    let target = offset.map(|x| x.max(0) as usize);
    let source: [usize; 3] = core::array::from_fn(|a| (target[a] as i64 - offset[a]) as usize);
    let key = |i: [usize; 3]| morton::from_index_and_level(i, level);
    (key(source), key(target))
}

/// n points inside `key`, away from its faces by a tenth of its half-width.
fn points_in(key: MortonKey, n: usize, rng: &mut Rng) -> Vec<[f64; 3]> {
    let domain = domain();
    let (c, r) = (
        geometry::centre(key, &domain),
        radius(morton::level(key) as usize, &domain),
    );
    (0..n)
        .map(|_| core::array::from_fn(|i| c[i] + 0.9 * r * rng.next()))
        .collect()
}

/// A leaf's source chunk of m charge vectors of `kind` at `points`.
fn source_chunk(
    d: &Densities,
    kind: ChargeKind,
    m: usize,
    leaf: MortonKey,
    points: &[[f64; 3]],
    rng: &mut Rng,
) -> Vec<f64> {
    let n = points.len();
    let planes: Vec<[Vec<f64>; 2]> = (0..m)
        .map(|_| {
            let re = (0..n).map(|_| rng.next()).collect();
            let im = match kind {
                ChargeKind::Real => vec![0.0; n],
                ChargeKind::Complex => (0..n).map(|_| rng.next()).collect(),
            };
            [re, im]
        })
        .collect();
    let q: Vec<Charges<'_, f64>> = planes
        .iter()
        .map(|[re, im]| match kind {
            ChargeKind::Real => Charges::Real(re),
            ChargeKind::Complex => Charges::Complex { re, im },
        })
        .collect();
    let mut chunk = vec![0.0; n * d.source_point_size()];
    for (u, &x) in chunk.as_chunks_mut::<3>().0.iter_mut().zip(points) {
        *u = geometry::leaf_coordinates(x, leaf, &domain());
    }
    let indices: Vec<usize> = (0..n).collect();
    d.write_leaf_charges(&q, &indices, &mut chunk[3 * n..]);
    chunk
}

/// The pair body on every V offset of levels 2 and 3 against Rotation's: the diagonal classes within
/// 2E(l, c) in CH1.3d1's measure (relative ℓ2 of the potential at points of the target box
/// from charges in the source box), each vector, E by the constants of the transfer functions'
/// build (the interpolated build, Helmholtz CH4.2c2, on the one-vector cases); the RCR classes
/// Rotation's bits.
#[test]
fn the_pair_body_against_rotation_within_the_bound() {
    use TransferBuild::{Direct, Interpolated};
    for (level, (re, im, kind, m, build)) in [2, 3].into_iter().flat_map(|level| {
        [
            (40.0, 0.0, ChargeKind::Complex, 1, Direct),
            (40.0, 0.0, ChargeKind::Real, 2, Direct),
            (40.0, 4.0, ChargeKind::Complex, 1, Direct),
            (40.0, 0.0, ChargeKind::Complex, 1, Interpolated),
            (40.0, 4.0, ChargeKind::Complex, 1, Interpolated),
        ]
        .map(|case| (level, case))
    }) {
        let k = Wavenumber::new(re, im).unwrap();
        let densities = Densities::new(k.kind(), kind, m);
        let mut diagonal = operator::<f64>(k, kind, m, HelmholtzDiagonal::Tolerance(EPSILON))
            .with_diagonal_build(build);
        let mut rotation = operator::<f64>(k, kind, m, HelmholtzDiagonal::Off);
        let p = diagonal.degree(level);
        let bounds = class_bounds(p, k_hat(&k, level), build.bound_constants(false));
        let stored = diagonal
            .diagonal_classes(level)
            .expect("level 2 is diagonal");
        let mut rng = Rng(11);
        let (mut worst, mut taken) = (0.0f64, 0);
        let n = 24;
        for d in 0..M2L_OFFSET_COUNT {
            let (source, target) = pair(level, d);
            let sources = points_in(source, n, &mut rng);
            let chunk = source_chunk(&densities, kind, m, source, &sources, &mut rng);
            let mut multipole = vec![0.0; rotation.multipole_size(level)];
            rotation.p2m(source, &chunk, &mut multipole);
            let (mut a, mut b) = (
                vec![0.0; rotation.local_size(level)],
                vec![0.0; rotation.local_size(level)],
            );
            diagonal.m2l(source, target, d, &multipole, &mut a);
            rotation.m2l(source, target, d, &multipole, &mut b);
            let class = class_index(m2l_offsets()[d]);
            if !stored.contains(&class) {
                assert_eq!(a, b, "an RCR class runs Rotation's body, offset {d}");
                continue;
            }
            taken += 1;
            let targets = points_in(target, n, &mut rng);
            let input: Vec<f64> = targets
                .iter()
                .flat_map(|&x| geometry::leaf_coordinates::<f64>(x, target, &domain()))
                .collect();
            let size = rotation.target_output_point_size();
            let (mut fa, mut fb) = (vec![0.0; n * size], vec![0.0; n * size]);
            rotation.l2p(target, &a, &input, &mut fa);
            rotation.l2p(target, &b, &input, &mut fb);
            // Each vector's potential, two columns (Re, Im) per vector, n values each.
            for v in 0..m {
                let column = |f: &[f64], c: usize| f[c * n..(c + 1) * n].to_vec();
                let (mut diff, mut norm) = (0.0, 0.0);
                for c in [2 * v, 2 * v + 1] {
                    for (x, y) in column(&fa, c).iter().zip(column(&fb, c)) {
                        diff += (x - y) * (x - y);
                        norm += y * y;
                    }
                }
                let error = (diff / norm).sqrt();
                let ratio = error / bounds[class];
                worst = worst.max(ratio);
                assert!(
                    ratio <= 2.0,
                    "k = {re} + {im}i, {build:?}, offset {d}, class {class}, vector {v}: {error:.2e} against E = {:.2e}",
                    bounds[class]
                );
            }
        }
        assert!(taken > 0);
        println!(
            "level {level}, k = {re} + {im}i, {kind:?}, m = {m}, {build:?}: {taken} offsets diagonal, worst {worst:.3} E"
        );
    }
}

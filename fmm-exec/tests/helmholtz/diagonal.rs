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
    HelmholtzDiagonal, HelmholtzM2l, HelmholtzOperator, TransferBuild, Wavenumber,
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

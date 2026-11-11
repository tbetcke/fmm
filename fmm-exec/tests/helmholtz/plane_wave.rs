//! The operator's exponential M2L (Helmholtz CH6.1d): each level's precision by the rule
//! and named, the levels it reports, and the pair body bit for bit the pair translation of
//! `nd_fmm_tables::helmholtz::plane_wave` along the direction merge and shift gives the
//! pair, for one and two multipole planes and two vectors; the split's setting (Helmholtz
//! CH6.2f); Auto's levels with waves (Helmholtz CH6.2h); the raised degrees (Helmholtz
//! CH6.1g) and where they apply (Helmholtz CH6.1h). The level call by merge and shift
//! is tested with the FMM in `tests/helmholtz_fmm.rs`.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    AUTO_PLANE_WAVE_MAX_IM_KAPPA, ChargeKind, DegreeRule, Densities, HelmholtzM2l,
    HelmholtzOperator, MIN_BUFFER_BOXES, PlaneWaveCounts, PlaneWavePrecision, PlaneWaveRaise,
    PlaneWaveSplit, Wavenumber, auto_batched_max_degree, auto_plane_wave_min_degree,
};
use nd_fmm_exec::operator::P2pChoice;
use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_simd::Isa;
use nd_fmm_tables::PlaneWaveTables;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::{MergeLevel, PlaneWaveM2l, select};
use nd_fmm_tables::plane_wave::MergeGeometry;
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

use Precision::{Iprec1, Iprec2, Iprec3};

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

/// Degrees the rule gives waves without a tolerance, and one above them.
const DEGREES: [usize; 6] = [0, 0, 6, 10, 15, 22];

fn operator(
    k: Wavenumber,
    kind: ChargeKind,
    m: usize,
    precision: PlaneWavePrecision,
) -> HelmholtzOperator<f64> {
    let densities = Densities::new(k.kind(), kind, m);
    HelmholtzOperator::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_plane_wave_precision(precision)
        .with_m2l(HelmholtzM2l::PlaneWave, 0)
}

fn precisions(op: &HelmholtzOperator<f64>) -> Vec<Option<Precision>> {
    (0..op.nlevels())
        .map(|l| op.plane_wave_precision(l))
        .collect()
}

/// A source and a target on `level` whose offset is the V-list offset d.
fn pair(level: usize, d: usize) -> (MortonKey, MortonKey) {
    let offset = m2l_offsets()[d];
    let target = offset.map(|x| x.max(0) as usize);
    let source: [usize; 3] = core::array::from_fn(|a| (target[a] as i64 - offset[a]) as usize);
    let key = |i: [usize; 3]| morton::from_index_and_level(i, level);
    (key(source), key(target))
}

#[test]
fn the_rule_chooses_each_levels_precision() {
    // k̂₀ = k r₀ = 10, so κ_l = 20 · 2⁻ˡ lies in the tables on every level.
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let op = operator(k, ChargeKind::Complex, 1, PlaneWavePrecision::Rule);
    let want = [None, None, Some(Iprec2), Some(Iprec2), Some(Iprec2), None];
    assert_eq!(precisions(&op), want);
    assert_eq!(
        precisions(&op),
        select(Complex::new(10.0, 0.0), &DEGREES, None)
    );
    assert_eq!(op.m2l_choice(), HelmholtzM2l::PlaneWave);
    for (level, precision) in want.iter().enumerate().skip(2) {
        let on = match precision {
            Some(_) => HelmholtzM2l::PlaneWave,
            None => HelmholtzM2l::Rotation,
        };
        assert_eq!(op.m2l_on(level), on, "level {level}");
    }
    assert!(op.plane_wave_bytes() > 0);
    println!(
        "{:.1} MB for levels 2–4 and one scratch set",
        op.plane_wave_bytes() as f64 / 1e6
    );
    // Rotation holds nothing of it.
    let rotation = operator(k, ChargeKind::Complex, 1, PlaneWavePrecision::Rule)
        .with_m2l(HelmholtzM2l::Rotation, 0);
    assert_eq!(precisions(&rotation), [None; 6]);
    assert_eq!(rotation.plane_wave_bytes(), 0);
}

/// The waves buffer (CH6.2a): sized from the plan's boxes per level and the bound, at least
/// `MIN_BUFFER_BOXES` boxes of a level (or the level's boxes), at most every box of it, in
/// either order of the calls; counted in `plane_wave_bytes`; nothing under Rotation.
#[test]
fn the_waves_buffer_is_bounded() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let boxes = [1, 8, 64, 512, 4096, 32768];
    let op = || {
        operator(k, ChargeKind::Complex, 1, PlaneWavePrecision::Rule).with_plane_wave_boxes(&boxes)
    };
    // The reals of one box's up and down waves on each level with waves (levels 2–4).
    let precisions = select(Complex::new(10.0, 0.0), &DEGREES, None);
    let m2l = PlaneWaveM2l::<f64>::build(Complex::new(10.0, 0.0), &DEGREES, &precisions);
    let box_waves: Vec<Option<usize>> = (0..DEGREES.len())
        .map(|l| m2l.level(l).map(|level| 2 * level.tables().wave_len()))
        .collect();
    // Each level's boxes in the buffer at a bound: as many as fit, at least the floor, at
    // most the level's; the buffer and the source list the largest of them, the positions
    // the largest level's boxes, the rest (tables and scratch) the same at every bound.
    let capacity = |bound: Option<usize>, l: usize| {
        let w = box_waves[l].map_or(0, |w| w * size_of::<f64>());
        bound
            .map_or(boxes[l], |b| (b / w).max(MIN_BUFFER_BOXES))
            .min(boxes[l])
    };
    let largest = |f: &dyn Fn(usize) -> usize| {
        (0..DEGREES.len())
            .filter(|&l| box_waves[l].is_some())
            .map(f)
            .max()
            .unwrap()
    };
    let buffer =
        |bound| largest(&|l| capacity(bound, l) * box_waves[l].unwrap() * size_of::<f64>());
    let held =
        |bound| buffer(bound) + 4 * (largest(&|l| capacity(bound, l)) + largest(&|l| boxes[l]));
    let whole = buffer(None);
    let floor = largest(&|l| boxes[l].min(MIN_BUFFER_BOXES) * box_waves[l].unwrap() * 8);

    let unbounded = op();
    let rest = unbounded.plane_wave_bytes() - held(None);
    assert_eq!(unbounded.plane_wave_buffer(), None);
    assert_eq!(unbounded.plane_wave_buffer_bytes(), whole);
    assert!(rest >= m2l.bytes());
    assert_eq!(unbounded.plane_wave_counts(), PlaneWaveCounts::default());
    for bound in [
        0,
        1,
        floor / 2,
        floor,
        whole / 5,
        whole / 2,
        whole,
        usize::MAX,
    ] {
        let op = op().with_plane_wave_buffer(Some(bound));
        assert_eq!(op.plane_wave_buffer(), Some(bound));
        assert_eq!(
            op.plane_wave_buffer_bytes(),
            buffer(Some(bound)),
            "bound {bound}"
        );
        assert_eq!(
            op.plane_wave_bytes(),
            rest + held(Some(bound)),
            "bound {bound}"
        );
        assert!(op.plane_wave_buffer_bytes() >= floor);
        assert!(op.plane_wave_buffer_bytes() <= bound.max(floor).min(whole));
    }
    let middle = whole / 5;
    let op = op().with_plane_wave_buffer(Some(middle));
    // The order of the calls does not matter.
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let before = HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_plane_wave_buffer(Some(middle))
        .with_plane_wave_boxes(&boxes)
        .with_m2l(HelmholtzM2l::PlaneWave, 0);
    assert_eq!(
        before.plane_wave_buffer_bytes(),
        op.plane_wave_buffer_bytes()
    );
    assert_eq!(before.plane_wave_bytes(), op.plane_wave_bytes());
    // Rotation holds none of it.
    let rotation = op.with_m2l(HelmholtzM2l::Rotation, 0);
    assert_eq!(rotation.plane_wave_buffer_bytes(), 0);
    assert_eq!(rotation.plane_wave_bytes(), 0);
}

#[test]
fn the_rule_reads_the_tolerance() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    for (tolerance, want) in [
        (1e-3, Some(Iprec2)),
        (1e-6, Some(Iprec2)),
        (1e-7, Some(Iprec3)),
        (1e-9, None),
    ] {
        let mut expected = vec![want; 6];
        expected[..2].fill(None);
        // Before and after `with_m2l`, the same.
        let op = operator(k, ChargeKind::Complex, 1, PlaneWavePrecision::Rule)
            .with_plane_wave_tolerance(Some(tolerance));
        assert_eq!(precisions(&op), expected, "ε = {tolerance:e}");
        let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
        let before = HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
            .unwrap()
            .with_plane_wave_tolerance(Some(tolerance))
            .with_m2l(HelmholtzM2l::PlaneWave, 0);
        assert_eq!(precisions(&before), expected);
        assert_eq!(
            expected,
            select(Complex::new(10.0, 0.0), &DEGREES, Some(tolerance))
        );
        // Named precisions ignore it.
        let named = op.with_plane_wave_precision(PlaneWavePrecision::PerLevel(vec![
            None,
            None,
            Some(Iprec1),
        ]));
        assert_eq!(
            precisions(&named),
            [None, None, Some(Iprec1), None, None, None]
        );
    }
    // Back to no tolerance: the degrees.
    let op = operator(k, ChargeKind::Complex, 1, PlaneWavePrecision::Rule)
        .with_plane_wave_tolerance(Some(1e-8))
        .with_plane_wave_tolerance(None);
    assert_eq!(precisions(&op)[4], Some(Iprec2));
    assert_eq!(
        precisions(&op),
        select(Complex::new(10.0, 0.0), &DEGREES, None)
    );
}

/// The degrees an operator runs at, level by level.
fn degrees(op: &HelmholtzOperator<f64>) -> Vec<usize> {
    (0..op.nlevels()).map(|l| op.degree(l)).collect()
}

/// Below 10⁻⁶ the rule raises every level with a quadrature to the resolving degree
/// (Helmholtz CH6.1g; CH6.1f's candidate 4), the store sizes and every operator of the level
/// with it; a later setting without those waves takes the degrees given back. Named
/// precisions and named degrees raise nothing; Auto takes the raised levels at 10⁻⁸ and keeps
/// D8's bound in 10⁻⁷ ≤ ε < 10⁻⁶.
#[test]
fn the_rule_raises_the_degrees_below_the_level_degrees_tolerance() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let given =
        || HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8).unwrap();
    let sizes = |op: &HelmholtzOperator<f64>| -> Vec<(usize, usize)> {
        (0..op.nlevels())
            .map(|l| (op.multipole_size(l), op.local_size(l)))
            .collect()
    };
    let given_sizes = sizes(&given());
    for (tolerance, precision, d) in [(1e-6, Iprec2, 0), (1e-7, Iprec3, 28), (1e-8, Iprec3, 32)] {
        let raised: Vec<usize> = DEGREES
            .iter()
            .enumerate()
            .map(|(l, &p)| if l >= 2 { p.max(d) } else { p })
            .collect();
        let op = given()
            .with_plane_wave_tolerance(Some(tolerance))
            .with_m2l(HelmholtzM2l::PlaneWave, 0);
        let mut want = vec![Some(precision); DEGREES.len()];
        want[..2].fill(None);
        assert_eq!(precisions(&op), want, "ε = {tolerance:e}");
        assert_eq!(degrees(&op), raised, "ε = {tolerance:e}");
        let base: Vec<usize> = (0..op.nlevels()).map(|l| op.base_degree(l)).collect();
        assert_eq!(base, DEGREES);
        // The stores and the tables of the raised degrees: an operator named at them.
        let named =
            HelmholtzOperator::<f64>::new(k, densities, &domain(), &raised, false, 8).unwrap();
        assert_eq!(sizes(&op), sizes(&named), "ε = {tolerance:e}");
        for (l, &p) in raised.iter().enumerate() {
            assert_eq!(op.tables().degree(l), p, "level {l}");
        }
        // The order of the calls does not matter.
        let after = given()
            .with_m2l(HelmholtzM2l::PlaneWave, 0)
            .with_plane_wave_tolerance(Some(tolerance));
        assert_eq!(degrees(&after), raised);
        assert_eq!(precisions(&after), want);
        // Rotation afterwards: the degrees given, and their stores.
        let rotation = op.with_m2l(HelmholtzM2l::Rotation, 0);
        assert_eq!(degrees(&rotation), DEGREES);
        assert_eq!(sizes(&rotation), given_sizes);
        // Named precisions raise nothing.
        let named = rotation
            .with_m2l(HelmholtzM2l::PlaneWave, 0)
            .with_plane_wave_precision(PlaneWavePrecision::PerLevel(vec![Some(precision); 6]));
        assert_eq!(degrees(&named), DEGREES);
        assert_eq!(precisions(&named), want);
        // Auto (f64, the exponential M2L from degree 0) keeps D8's bound in
        // 10⁻⁷ ≤ ε < 10⁻⁶ and takes the rule's levels, raised, elsewhere (signed off
        // 2026-11-11).
        let auto = given()
            .with_plane_wave_tolerance(Some(tolerance))
            .with_m2l_plane_wave_min_degree(Some(0))
            .with_m2l(HelmholtzM2l::Auto, 0);
        let (auto_degrees, auto_want) = if tolerance == 1e-7 {
            (DEGREES.to_vec(), vec![None; 6])
        } else {
            (raised.clone(), want.clone())
        };
        assert_eq!(degrees(&auto), auto_degrees, "Auto at ε = {tolerance:e}");
        assert_eq!(precisions(&auto), auto_want, "Auto at ε = {tolerance:e}");
    }
    // Without a tolerance (named degrees) nothing is raised.
    let op = given().with_m2l(HelmholtzM2l::PlaneWave, 0);
    assert_eq!(degrees(&op), DEGREES);
    // And below the rule's tolerances no level has waves, none raised.
    let op = given()
        .with_plane_wave_tolerance(Some(1e-9))
        .with_m2l(HelmholtzM2l::PlaneWave, 0);
    assert_eq!(degrees(&op), DEGREES);
    assert_eq!(precisions(&op), vec![None; 6]);
}

/// The translation degrees an operator runs at, level by level.
fn translation_degrees(op: &HelmholtzOperator<f64>) -> Vec<usize> {
    (0..op.nlevels())
        .map(|l| op.translation_degree(l))
        .collect()
}

/// [`PlaneWaveRaise::Leaves`] and [`PlaneWaveRaise::M2l`] (Helmholtz CH6.1h): the raised
/// levels' stores, M2L and tables at the resolving degree as under [`PlaneWaveRaise::Level`]
/// (the default), their translations to and from the parent at the degrees given; the M2L
/// pair body the same bits under every form; the leaves' P2M and L2P `Level`'s under
/// `Leaves` and, under `M2l`, Rotation's at the degrees given on the first degrees, the
/// multipole's others zero; the setting kept in either order of the calls, Rotation
/// afterwards back at the degrees given, Auto alike, and nothing raised at 10⁻⁶.
#[test]
fn the_raised_levels_translations_keep_the_degrees_given() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let given =
        || HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8).unwrap();
    let sizes = |op: &HelmholtzOperator<f64>| -> Vec<(usize, usize)> {
        (0..op.nlevels())
            .map(|l| (op.multipole_size(l), op.local_size(l)))
            .collect()
    };
    assert_eq!(given().plane_wave_raise(), PlaneWaveRaise::Level);
    assert_eq!(PlaneWaveRaise::default(), PlaneWaveRaise::Level);
    let mut rotation = given().with_m2l(HelmholtzM2l::Rotation, 0);
    // A leaf of level 4 (degree 15 given) with 8 sources and targets in its units.
    let leaf = morton::from_index_and_level([3, 5, 6], 4);
    let mut rng = Rng(5);
    let n = 8;
    let mut chunk: Vec<f64> = (0..3 * n).map(|_| 0.9 * rng.next()).collect();
    let charges: Vec<[f64; 2]> = (0..n).map(|_| [rng.next(), rng.next()]).collect();
    for (j, &[re, im]) in charges.iter().enumerate() {
        chunk.extend([re, im]);
        assert_eq!(densities.charge_index(n, j, 0, 1), 3 * n + 2 * j + 1);
    }
    let targets: Vec<f64> = (0..3 * n).map(|_| 0.9 * rng.next()).collect();
    for (tolerance, d) in [(1e-6, 0), (1e-7, 28), (1e-8, 32)] {
        let raised: Vec<usize> = DEGREES
            .iter()
            .enumerate()
            .map(|(l, &p)| if l >= 2 { p.max(d) } else { p })
            .collect();
        let plane_wave = |raise| {
            given()
                .with_plane_wave_tolerance(Some(tolerance))
                .with_plane_wave_raise(raise)
                .with_m2l(HelmholtzM2l::PlaneWave, 0)
        };
        let mut level = plane_wave(PlaneWaveRaise::Level);
        assert_eq!(degrees(&level), raised, "ε = {tolerance:e}");
        assert_eq!(translation_degrees(&level), raised, "ε = {tolerance:e}");
        // The default is `Level`.
        let default = given()
            .with_plane_wave_tolerance(Some(tolerance))
            .with_m2l(HelmholtzM2l::PlaneWave, 0);
        assert_eq!(translation_degrees(&default), raised);
        let multipole_size = level.multipole_size(4);
        let local: Vec<f64> = (0..level.local_size(4)).map(|_| rng.next()).collect();
        let mut level_multipole = vec![0.0; multipole_size];
        level.p2m(leaf, &chunk, &mut level_multipole);
        let mut level_output = vec![0.0; level.target_output_point_size() * n];
        level.l2p(leaf, &local, &targets, &mut level_output);
        for raise in [PlaneWaveRaise::Leaves, PlaneWaveRaise::M2l] {
            let what = format!("{raise:?} at ε = {tolerance:e}");
            let mut op = plane_wave(raise);
            assert_eq!(op.plane_wave_raise(), raise);
            assert_eq!(degrees(&op), raised, "{what}");
            assert_eq!(translation_degrees(&op), DEGREES, "{what}");
            assert_eq!(sizes(&op), sizes(&level), "{what}");
            assert_eq!(precisions(&op), precisions(&level), "{what}");
            for (l, (&p, &q)) in DEGREES.iter().zip(&raised).enumerate() {
                assert_eq!(op.tables().degree(l), p, "{what}, level {l}");
                assert_eq!(op.tables().m2l_degree(l), q, "{what}, level {l}");
            }
            // The M2L of a raised level reads and fills whole expansions at p_w: the same
            // bits.
            for level_index in 2..DEGREES.len() {
                let multipole: Vec<f64> = (0..op.multipole_size(level_index))
                    .map(|_| rng.next())
                    .collect();
                let start: Vec<f64> = (0..op.local_size(level_index))
                    .map(|_| rng.next())
                    .collect();
                let (source, target) = pair(level_index, 100);
                let (mut a, mut b) = (start.clone(), start);
                level.m2l(source, target, 100, &multipole, &mut a);
                op.m2l(source, target, 100, &multipole, &mut b);
                assert!(a == b, "{what}, level {level_index}: M2L");
            }
            // The leaf's P2M and L2P: `Level`'s, or under `M2l` Rotation's at the degree
            // given on the first degrees.
            let mut multipole = vec![0.0; multipole_size];
            op.p2m(leaf, &chunk, &mut multipole);
            let mut output = vec![0.0; level_output.len()];
            op.l2p(leaf, &local, &targets, &mut output);
            let (q, p) = (raised[4], DEGREES[4]);
            if raise == PlaneWaveRaise::Leaves || q == p {
                assert!(multipole == level_multipole, "{what}: P2M");
                assert!(output == level_output, "{what}: L2P");
            } else {
                let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
                let mut want = vec![0.0; rotation.multipole_size(4)];
                rotation.p2m(leaf, &chunk, &mut want);
                for (plane, want) in want.chunks_exact(len).enumerate() {
                    let got = &multipole[plane * stride..(plane + 1) * stride];
                    assert!(got[..len] == *want, "{what}: P2M's first degrees");
                    assert!(got[len..].iter().all(|&x| x == 0.0), "{what}: P2M above p");
                }
                let first: Vec<f64> = local
                    .chunks_exact(stride)
                    .flat_map(|plane| plane[..len].iter().copied())
                    .collect();
                let mut want = vec![0.0; output.len()];
                rotation.l2p(leaf, &first, &targets, &mut want);
                assert!(output == want, "{what}: L2P on the first degrees");
            }
            // Named first, then the M2L chosen: the same; Rotation afterwards: the degrees
            // given.
            let after = given()
                .with_m2l(HelmholtzM2l::PlaneWave, 0)
                .with_plane_wave_tolerance(Some(tolerance))
                .with_plane_wave_raise(raise);
            assert_eq!(degrees(&after), raised, "{what}");
            assert_eq!(translation_degrees(&after), DEGREES, "{what}");
            let back = op.with_m2l(HelmholtzM2l::Rotation, 0);
            assert_eq!(degrees(&back), DEGREES, "{what}");
            assert_eq!(translation_degrees(&back), DEGREES, "{what}");
            assert_eq!(back.plane_wave_raise(), raise, "{what}");
            // Auto at 10⁻⁸ raises the same levels with the translations kept.
            let auto = given()
                .with_plane_wave_tolerance(Some(tolerance))
                .with_plane_wave_raise(raise)
                .with_m2l_plane_wave_min_degree(Some(0))
                .with_m2l(HelmholtzM2l::Auto, 0);
            let auto_raised = if tolerance == 1e-7 {
                &DEGREES[..]
            } else {
                &raised[..]
            };
            assert_eq!(degrees(&auto), auto_raised, "Auto, {what}");
            assert_eq!(translation_degrees(&auto), DEGREES, "Auto, {what}");
        }
    }
}

/// The leaf operators of a raised level (Helmholtz CH6.1h), at ε = 10⁻⁸ on level 4 (15 → 32),
/// with one multipole plane (real k, real charges) and two (complex k, complex charges): under
/// [`PlaneWaveRaise::Leaves`] P2M, L2P, P2L and M2P are `Level`'s bit for bit; under
/// [`PlaneWaveRaise::M2l`] Rotation's at the degree given on the first degrees, P2M's and
/// P2L's others zero.
#[test]
fn the_raised_levels_leaf_operators_by_the_form() {
    // The target leaf on level 4, a W box on level 5 (a child of a neighbour, not adjacent),
    // an X leaf on level 3 (a neighbour of the parent, not adjacent).
    let leaf = morton::from_index_and_level([3, 5, 6], 4);
    let w_box = morton::from_index_and_level([9, 10, 12], 5);
    let x_leaf = morton::from_index_and_level([0, 2, 3], 3);
    for (k, kind) in [
        (Wavenumber::new(20.0, 0.0).unwrap(), ChargeKind::Real),
        (Wavenumber::new(20.0, 2.0).unwrap(), ChargeKind::Complex),
    ] {
        let densities = Densities::new(k.kind(), kind, 1);
        let planes = densities.multipole_planes();
        let op = |m2l, raise| {
            HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
                .unwrap()
                .with_plane_wave_tolerance(Some(1e-8))
                .with_plane_wave_raise(raise)
                .with_m2l(m2l, 0)
        };
        let mut rotation = op(HelmholtzM2l::Rotation, PlaneWaveRaise::Level);
        let mut level = op(HelmholtzM2l::PlaneWave, PlaneWaveRaise::Level);
        let mut leaves = op(HelmholtzM2l::PlaneWave, PlaneWaveRaise::Leaves);
        let mut m2l = op(HelmholtzM2l::PlaneWave, PlaneWaveRaise::M2l);
        assert_eq!((level.degree(4), level.degree(5)), (32, 32), "{kind:?}");
        assert_eq!(m2l.translation_degree(4), 15, "{kind:?}");
        // 8 sources and targets in the leaf's units; the charges after the positions.
        let mut rng = Rng(9);
        let n = 8;
        let columns = densities.source_point_size() - 3;
        let chunk: Vec<f64> = (0..(3 + columns) * n)
            .map(|i| {
                if i < 3 * n {
                    0.9 * rng.next()
                } else {
                    rng.next()
                }
            })
            .collect();
        let targets: Vec<f64> = (0..3 * n).map(|_| 0.9 * rng.next()).collect();
        let output_len = level.target_output_point_size() * n;
        type Ops<'a> = [&'a mut HelmholtzOperator<f64>; 3];
        let what = |name: &str| format!("{kind:?}: {name}");
        // P2M into the leaf, P2L from the X leaf into the leaf as a box.
        for (name, source) in [("P2M", None), ("P2L", Some(x_leaf))] {
            let run = |op: &mut HelmholtzOperator<f64>| {
                let size = match source {
                    None => op.multipole_size(4),
                    Some(_) => op.local_size(4),
                };
                let mut out = vec![0.0; size];
                match source {
                    None => op.p2m(leaf, &chunk, &mut out),
                    Some(s) => op.p2l(s, leaf, &chunk, &mut out),
                }
                out
            };
            let [a, b, c]: Ops = [&mut level, &mut leaves, &mut m2l];
            let (at_level, at_leaves, at_m2l) = (run(a), run(b), run(c));
            assert!(at_leaves == at_level, "{}", what(name));
            let want = run(&mut rotation);
            let plane_count = if source.is_some() { 2 } else { planes };
            check_first_degrees(&at_m2l, &want, plane_count, 15, 32, &what(name));
        }
        // L2P from the leaf's local, M2P from the W box's multipole.
        let local: Vec<f64> = (0..level.local_size(4)).map(|_| rng.next()).collect();
        let multipole: Vec<f64> = (0..level.multipole_size(5)).map(|_| rng.next()).collect();
        for (name, from, plane_count, p) in
            [("L2P", &local, 2, 15), ("M2P", &multipole, planes, 22)]
        {
            let run = |op: &mut HelmholtzOperator<f64>, input: &[f64]| {
                let mut out = vec![0.0; output_len];
                match name {
                    "L2P" => op.l2p(leaf, input, &targets, &mut out),
                    _ => op.m2p(w_box, leaf, input, &targets, &mut out),
                }
                out
            };
            let at_level = run(&mut level, from);
            assert!(run(&mut leaves, from) == at_level, "{}", what(name));
            let first = first_degrees(from, plane_count, p, 32);
            assert!(
                run(&mut m2l, from) == run(&mut rotation, &first),
                "{}: Rotation's on the first degrees",
                what(name)
            );
        }
    }
}

/// The degrees ≤ p of `planes` planes (each vector's, one after the other) of degree q.
fn first_degrees(block: &[f64], planes: usize, p: usize, q: usize) -> Vec<f64> {
    let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
    assert_eq!(block.len() % (planes * stride), 0);
    block
        .chunks_exact(stride)
        .flat_map(|plane| plane[..len].iter().copied())
        .collect()
}

/// `got`, planes of degree q, holds `want`'s planes of degree p in its first degrees and zero
/// above.
fn check_first_degrees(got: &[f64], want: &[f64], planes: usize, p: usize, q: usize, what: &str) {
    let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
    assert_eq!(got.len() / stride, want.len() / len, "{what}: planes");
    assert_eq!(got.len() % (planes * stride), 0, "{what}: planes");
    for (got, want) in got.chunks_exact(stride).zip(want.chunks_exact(len)) {
        assert!(got[..len] == *want, "{what}: the first degrees");
        assert!(
            got[len..].iter().all(|&x| x == 0.0),
            "{what}: above the degree given"
        );
    }
}

/// The rule's largest degree without a tolerance is the calibrated plateau at its tolerance.
#[test]
fn the_rules_degree_is_the_calibrated_plateau() {
    use nd_fmm_tables::helmholtz::plane_wave::merge::{
        MAX_PLANE_WAVE_DEGREE, PLANE_WAVE_MIN_TOLERANCE,
    };
    assert_eq!(
        DegreeRule::CALIBRATED.plateau(PLANE_WAVE_MIN_TOLERANCE),
        MAX_PLANE_WAVE_DEGREE
    );
}

#[test]
fn named_precisions() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    // Levels 0 and 1 ignored, levels beyond the list none.
    let named = PlaneWavePrecision::PerLevel(vec![Some(Iprec3), Some(Iprec3), Some(Iprec3)]);
    let op = operator(k, ChargeKind::Real, 1, named.clone());
    assert_eq!(
        precisions(&op),
        [None, None, Some(Iprec3), None, None, None]
    );
    assert_eq!(op.m2l_on(3), HelmholtzM2l::Rotation);
    // Named after `with_m2l`, the same.
    let densities = Densities::new(k.kind(), ChargeKind::Real, 1);
    let after = HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_m2l(HelmholtzM2l::PlaneWave, 0)
        .with_plane_wave_precision(named);
    assert_eq!(precisions(&after), precisions(&op));
    // κ₂ = 2 k r₂ = 100 lies outside the tables (Re κ < 16π): RCR there whatever is named.
    let far = Wavenumber::new(400.0, 0.0).unwrap();
    let all = PlaneWavePrecision::PerLevel(vec![Some(Iprec1); 6]);
    let op = operator(far, ChargeKind::Real, 1, all);
    assert_eq!(
        precisions(&op),
        [None, None, None, Some(Iprec1), Some(Iprec1), Some(Iprec1)]
    );
}

#[test]
fn the_split_is_a_setting() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    let op = operator(k, ChargeKind::Real, 1, PlaneWavePrecision::Rule);
    assert_eq!(op.plane_wave_split(), PlaneWaveSplit::Auto);
    assert_eq!(PlaneWaveSplit::default(), PlaneWaveSplit::Auto);
    // Kept through a later choice of the tables, and before `with_m2l` alike.
    for split in [PlaneWaveSplit::Parents, PlaneWaveSplit::Children] {
        let op = operator(k, ChargeKind::Real, 1, PlaneWavePrecision::Rule)
            .with_plane_wave_split(split)
            .with_plane_wave_reach(5.0);
        assert_eq!(op.plane_wave_split(), split);
        let densities = Densities::new(k.kind(), ChargeKind::Real, 1);
        let before = HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
            .unwrap()
            .with_plane_wave_split(split)
            .with_m2l(HelmholtzM2l::PlaneWave, 0);
        assert_eq!(before.plane_wave_split(), split);
        assert_eq!(before.plane_wave_counts(), PlaneWaveCounts::default());
    }
}

#[test]
fn the_pair_body_is_the_pair_translation() {
    let geometry = MergeGeometry::new();
    // One plane (real k, real charges) and two (complex k).
    for (re, im, kind) in [
        (20.0, 0.0, ChargeKind::Real),
        (20.0, 3.0, ChargeKind::Complex),
    ] {
        let k = Wavenumber::new(re, im).unwrap();
        let m = 2;
        let mut op = operator(k, kind, m, PlaneWavePrecision::Rule);
        let planes = op.densities().multipole_planes();
        let mut rng = Rng(7);
        for (level, &p) in DEGREES.iter().enumerate().take(5).skip(2) {
            let n = Layout::new(p).len();
            let precision = op.plane_wave_precision(level).expect("a level with waves");
            let k_hat = Complex::new(re * 0.5, im * 0.5) * 0.5f64.powi(level as i32);
            let merge = MergeLevel::<f64>::new(k_hat, p, precision, &geometry).unwrap();
            let mut scratch = merge.tables().scratch();
            let multipole: Vec<f64> = (0..op.multipole_size(level)).map(|_| rng.next()).collect();
            // Every offset in debug builds of the finest level, every fourth elsewhere.
            let step = if cfg!(debug_assertions) && level < 4 {
                4
            } else {
                1
            };
            for d in (0..M2L_OFFSET_COUNT).step_by(step) {
                let (source, target) = pair(level, d);
                let start: Vec<f64> = (0..op.local_size(level)).map(|_| rng.next()).collect();
                let mut got = start.clone();
                op.m2l(source, target, d, &multipole, &mut got);
                let direction =
                    PlaneWaveTables::<f64>::pair_direction(d, morton::child_index(target));
                let mut want = start;
                for v in 0..m {
                    merge.tables().m2l(
                        d,
                        direction,
                        &multipole[v * planes * n..(v + 1) * planes * n],
                        planes,
                        &mut want[v * 2 * n..(v + 1) * 2 * n],
                        &mut scratch,
                    );
                }
                assert!(
                    got == want,
                    "k = {re} + {im}i, level {level}, d = {d}: not the pair translation"
                );
            }
        }
    }
}

/// Auto with the exponential M2L (Helmholtz CH6.2h) on `DEGREES` at k, the crossover `d_b`
/// and the smallest degree `d_pw` named, at `tolerance`.
fn auto(
    k: Wavenumber,
    tolerance: Option<f64>,
    d_b: usize,
    d_pw: Option<usize>,
    on: bool,
) -> HelmholtzOperator<f64> {
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    HelmholtzOperator::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_plane_wave_tolerance(tolerance)
        .with_m2l_max_degree(Some(d_b))
        .with_m2l_plane_wave_min_degree(d_pw)
        .with_auto_plane_wave(on)
        .with_m2l(HelmholtzM2l::Auto, usize::MAX)
}

fn methods(op: &HelmholtzOperator<f64>) -> Vec<HelmholtzM2l> {
    (2..op.nlevels()).map(|l| op.m2l_on(l)).collect()
}

/// Auto's rule with waves: the exponential M2L from the smallest degree where the precision
/// rule gives a quadrature (ε 10⁻³ and 10⁻⁶; at 10⁻⁸ at the resolving degree, Helmholtz
/// CH6.1g), Batched at or below the crossover on the others, RCR else; off, with no
/// quadrature (ε below 10⁻⁸), in D8's kept band 10⁻⁷ ≤ ε < 10⁻⁶, or with the smallest
/// degree above every level's, Auto's levels and tables as before.
#[test]
fn auto_takes_the_waves_by_its_rule() {
    use HelmholtzM2l::{Batched, PlaneWave, Rotation};
    // κ_l = 20 · 2⁻ˡ in the tables on every level; DEGREES 6, 10, 15, 22 on levels 2–5
    // (crossover 6: Batched's tables at p ≥ 10 take seconds to build in debug).
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    // CH6.1e's rule gives iprec 2 at both tolerances (the same waves).
    for tolerance in [1e-3, 1e-6] {
        for (d_b, d_pw, want) in [
            (6, 10, [Batched, PlaneWave, PlaneWave, PlaneWave]),
            (6, 16, [Batched, Rotation, Rotation, PlaneWave]),
            (6, 15, [Batched, Rotation, PlaneWave, PlaneWave]),
            // The exponential M2L first where both apply.
            (6, 0, [PlaneWave; 4]),
            (6, 23, [Batched, Rotation, Rotation, Rotation]),
        ] {
            let case = format!("ε {tolerance:e}, d_b {d_b}, d_pw {d_pw}");
            let op = auto(k, Some(tolerance), d_b, Some(d_pw), true);
            assert_eq!(methods(&op), want, "{case}");
            assert_eq!(op.m2l_choice(), HelmholtzM2l::Auto);
            assert_eq!(op.plane_wave_min_degree(), Some(d_pw));
            assert_eq!(op.max_degree(), Some(d_b));
            // The rule's precisions on the levels with waves.
            let rule = select(Complex::new(10.0, 0.0), &DEGREES, Some(tolerance));
            for (level, &on) in want.iter().enumerate() {
                let expected = (on == PlaneWave).then_some(rule[level + 2]).flatten();
                assert_eq!(op.plane_wave_precision(level + 2), expected, "{case}");
            }
            // The levels with waves take no batched tables.
            let batched = want.iter().filter(|&&m| m == Batched).count();
            assert_eq!(op.batched_bytes() > 0, batched > 0, "{case}");
            assert_eq!(
                op.plane_wave_bytes() > 0,
                want.contains(&PlaneWave),
                "{case}"
            );
        }
    }
    // Off, below the rule's tolerances, or above every degree: Auto as before, its tables
    // included.
    let before = auto(k, Some(1e-6), 6, Some(10), false);
    let want: Vec<HelmholtzM2l> = DEGREES[2..]
        .iter()
        .map(|&p| if p <= 6 { Batched } else { Rotation })
        .collect();
    assert_eq!(methods(&before), want);
    for (case, op) in [
        ("ε 1e-7", auto(k, Some(1e-7), 6, Some(10), true)),
        ("ε 1e-9", auto(k, Some(1e-9), 6, Some(10), true)),
        ("d_pw 23", auto(k, Some(1e-6), 6, Some(23), true)),
    ] {
        assert_eq!(methods(&op), want, "{case}");
        assert_eq!(degrees(&op), DEGREES, "{case}");
        assert_eq!(op.batched_bytes(), before.batched_bytes(), "{case}");
        assert_eq!(op.plane_wave_bytes(), 0, "{case}");
    }
    assert_eq!(before.plane_wave_bytes(), 0);
    // At 10⁻⁸ (Helmholtz CH6.1g, D8's bound lifted there): iprec 3 on the levels of degree
    // at least d_pw, each raised to max(p_l, 32); Batched on level 2 at its own degree.
    let op = auto(k, Some(1e-8), 6, Some(10), true);
    assert_eq!(methods(&op), [Batched, PlaneWave, PlaneWave, PlaneWave]);
    assert_eq!(degrees(&op), [0, 0, 6, 32, 32, 32]);
    assert_eq!(
        precisions(&op),
        [None, None, None, Some(Iprec3), Some(Iprec3), Some(Iprec3)]
    );
    assert!(op.batched_bytes() > 0);
    // Off, the smallest degree is not reported; the setting before or after `with_m2l`.
    assert_eq!(
        auto(k, Some(1e-6), 6, None, false).plane_wave_min_degree(),
        None
    );
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let after = HelmholtzOperator::<f64>::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_m2l(HelmholtzM2l::Auto, usize::MAX)
        .with_m2l_max_degree(Some(6))
        .with_plane_wave_tolerance(Some(1e-6))
        .with_auto_plane_wave(true)
        .with_m2l_plane_wave_min_degree(Some(10));
    assert_eq!(
        methods(&after),
        methods(&auto(k, Some(1e-6), 6, Some(10), true))
    );
    // Another setting ignores it.
    let rotation = after.with_m2l(HelmholtzM2l::Rotation, 0);
    assert_eq!(methods(&rotation), [Rotation; 4]);
    assert_eq!(rotation.plane_wave_min_degree(), None);
}

/// The rule's smallest degree follows the kernel's ISA, as the crossover does; strongly
/// damped levels (Im κ_l from 5π) run without waves; f32 never takes them.
#[test]
fn auto_waves_by_the_machine_class_and_the_damping() {
    let k = Wavenumber::new(20.0, 0.0).unwrap();
    for isa in [Isa::Scalar, Isa::detect()] {
        let op = auto(k, Some(1e-6), 6, None, true)
            .with_m2l_max_degree(None)
            .with_p2p(P2pChoice::Isa(isa))
            .unwrap();
        assert_eq!(
            op.plane_wave_min_degree(),
            Some(auto_plane_wave_min_degree(isa, 2))
        );
        assert_eq!(op.max_degree(), Some(auto_batched_max_degree(isa, 2)));
    }
    // κ_l = (20 + 100i) 2⁻ˡ: Im κ₂ = 25 above 5π, Im κ₃ = 12.5 below; both in the tables.
    let damped = Wavenumber::new(20.0, 100.0).unwrap();
    let rule = select(Complex::new(10.0, 50.0), &DEGREES, Some(1e-6));
    assert!(rule[2].is_some() && rule[3].is_some());
    const { assert!(25.0 >= AUTO_PLANE_WAVE_MAX_IM_KAPPA && 12.5 < AUTO_PLANE_WAVE_MAX_IM_KAPPA) };
    let op = auto(damped, Some(1e-6), 0, Some(0), true);
    assert_eq!(op.m2l_on(2), HelmholtzM2l::Rotation);
    assert_eq!(op.m2l_on(3), HelmholtzM2l::PlaneWave);
    // PlaneWave itself keeps the level.
    let densities = Densities::new(damped.kind(), ChargeKind::Complex, 1);
    let named = HelmholtzOperator::<f64>::new(damped, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_plane_wave_tolerance(Some(1e-6))
        .with_m2l(HelmholtzM2l::PlaneWave, 0);
    assert_eq!(named.m2l_on(2), HelmholtzM2l::PlaneWave);
    // f32: none.
    let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
    let single = HelmholtzOperator::<f32>::new(k, densities, &domain(), &DEGREES, false, 8)
        .unwrap()
        .with_plane_wave_tolerance(Some(1e-6))
        .with_m2l_plane_wave_min_degree(Some(0))
        .with_auto_plane_wave(true)
        .with_m2l(HelmholtzM2l::Auto, usize::MAX);
    assert_eq!(single.plane_wave_min_degree(), None);
    assert!((2..single.nlevels()).all(|l| single.m2l_on(l) != HelmholtzM2l::PlaneWave));
}

/// The smallest degree is one above Batched's crossover, but 12 on AVX-512 with two planes.
#[test]
fn the_plane_wave_degrees_follow_the_crossovers() {
    for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon] {
        for planes in [1, 2] {
            let want = match (isa, planes) {
                (Isa::Avx512, 2) => 12,
                _ => auto_batched_max_degree(isa, planes) + 1,
            };
            assert_eq!(auto_plane_wave_min_degree(isa, planes), want, "{isa:?}");
            assert!(want <= auto_batched_max_degree(isa, planes) + 1);
        }
    }
}

//! The operator's exponential M2L (Helmholtz CH6.1d): each level's precision by the rule
//! and named, the levels it reports, and the pair body bit for bit the pair translation of
//! `nd_fmm_tables::helmholtz::plane_wave` along the direction merge and shift gives the
//! pair, for one and two multipole planes and two vectors; the split's setting (Helmholtz
//! CH6.2f). The level call by merge and shift is tested with the FMM in
//! `tests/helmholtz_fmm.rs`.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    ChargeKind, DegreeRule, Densities, HelmholtzM2l, HelmholtzOperator, MIN_BUFFER_BOXES,
    PlaneWaveCounts, PlaneWavePrecision, PlaneWaveSplit, Wavenumber,
};
use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
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
    for (tolerance, want) in [(1e-3, Some(Iprec2)), (1e-6, Some(Iprec2)), (1e-8, None)] {
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

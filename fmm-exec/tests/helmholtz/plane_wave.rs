//! The operator's exponential M2L (Helmholtz CH6.1d): each level's precision by the rule
//! and named, the levels it reports, and the pair body bit for bit the pair translation of
//! `nd_fmm_tables::helmholtz::plane_wave` along the direction merge and shift gives the
//! pair, for one and two multipole planes and two vectors. The level call by merge and
//! shift is tested with the FMM in `tests/helmholtz_fmm.rs`.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{
    ChargeKind, Densities, HelmholtzM2l, HelmholtzOperator, PlaneWavePrecision, Wavenumber,
};
use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::{FmmSizes, PairOperator};
use nd_fmm_tables::PlaneWaveTables;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::{MergeLevel, select};
use nd_fmm_tables::plane_wave::MergeGeometry;
use nd_octree::{MortonKey, PhysicalBox, morton};

use super::common::Rng;

use Precision::{Iprec1, Iprec2, Iprec3};

fn domain() -> Domain {
    Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap()
}

/// The degrees of the rule's three precisions and one above it.
const DEGREES: [usize; 6] = [0, 0, 6, 10, 14, 20];

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
    let want = [None, None, Some(Iprec1), Some(Iprec2), Some(Iprec3), None];
    assert_eq!(precisions(&op), want);
    assert_eq!(precisions(&op), select(Complex::new(10.0, 0.0), &DEGREES));
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

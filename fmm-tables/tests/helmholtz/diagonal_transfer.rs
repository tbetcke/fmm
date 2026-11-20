//! A level's transfer functions of the diagonal form and the pair (Helmholtz CH4.2a,
//! `nd_fmm_tables::helmholtz::diagonal::transfer`; proposal §3.6, §5.4;
//! `docs/helmholtz/tasks/CH4.2a.md`).
//!
//! Error measures:
//! - the maps, the stored samples against ref's `Transfer` times the factor, the permuted
//!   samples, the pair against its naive loop, the build in parts, the level at every chunk
//!   size: bit for bit (`to_bits`);
//! - an offset's samples against ref's `Transfer` of the offset itself: the largest absolute
//!   difference within 4 ε (f64::EPSILON, 2u) of the largest sample;
//! - the acceptance, an M2L against RCR: CH1.3d1's measure, the larger of the relative ℓ2
//!   errors of φ̃ and ∇φ̃ after `leaf::l2p` at points of the target box (uniform and its
//!   corners), the multipole by `leaf::p2m` of complex charges in the source box (uniform and
//!   its corners), within 2E = E/θ ("within E" as S6 fitted it, proposal §3.6). At E's floor
//!   (C₀ u) at large p f64 RCR rounds more than the diagonal form, so the comparison measures
//!   RCR's error too: where the measurement exceeds 2E, RCR's own error against a third path,
//!   ref's projection `direct::m2l`, is added to the limit (CH1.3d1 saw 1.99 E at k a = 54,
//!   p = 72; this task 2.4 E at p = 82 and 2.9 E at k a = 108, p = 131: the record).

use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::diagonal::{BoundConstants, Grid, THETA, Transfer, class_of, classes};
use nd_fmm_ref::helmholtz::{Level, Workspace, direct, leaf, rcr};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offset_index, m2l_offsets};
use nd_fmm_tables::helmholtz::diagonal::{
    CLASS_COUNT, Conversions, DiagonalM2l, LevelBuffers, TransferBuild, class_bounds,
    interpolation_steps, m2l_level,
};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_tables::parts::Serial;

use super::parts::Threads;

type C = Complex<f64>;

const ALL: [usize; CLASS_COUNT] = {
    let mut all = [0; CLASS_COUNT];
    let mut i = 0;
    while i < CLASS_COUNT {
        all[i] = i;
        i += 1;
    }
    all
};

/// Deterministic values in [−1, 1).
fn values(count: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

fn bits<T: RealScalar>(values: &[Complex<T>]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|c| (c.re.to_f64().to_bits(), c.im.to_f64().to_bits()))
        .collect()
}

fn real_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// k̂ with k a = `ka` on a level of half-width 1 (a = √3); `eta` = Im k / Re k.
fn k_hat(ka: f64, eta: f64) -> C {
    let a = 3f64.sqrt();
    C::new(ka / a, eta * ka / a)
}

/// The degree of proposal §3.5's formula (as `nd-fmm-ref`'s diagonal tests).
fn formula_degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.cbrt();
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2f64.ln();
    let big_p = (p_lo.powi(4) + p_hi.powi(4)).powf(0.25);
    big_p.ceil() as usize - 1
}

fn class_of_offset(o: [i32; 3]) -> usize {
    let rep = class_of(o).0;
    classes()
        .iter()
        .position(|c| c.representative == rep)
        .unwrap()
}

/// 4π ik e^{2β − Im(k)|t|}, as the tables form it.
fn factor(level: &Level<f64>, squared: i32) -> C {
    let k = level.k();
    let t = 2.0 * f64::from(squared).sqrt();
    let beta = level.damping();
    let c = 4.0 * core::f64::consts::PI;
    C::new(-(c * k.im), c * k.re) * (beta + beta - k.im * t).exp()
}

fn frame(o: [i32; 3]) -> Frame<f64> {
    Frame::new(o.map(|x| 2.0 * f64::from(x)), 1.0)
}

/// Every offset's map is ref's symmetry on every sample of the grid.
#[test]
fn the_maps_are_refs_symmetries() {
    for p in [3, 8] {
        let m2l = DiagonalM2l::<f64>::build(p, C::new(2.0, 0.0), &[]);
        let grid = Grid::<f64>::new(p);
        let (rings, n) = (grid.rings(), grid.azimuths());
        for (d, o) in m2l_offsets().iter().enumerate() {
            let o = o.map(|x| x as i32);
            let (_, symmetry) = class_of(o);
            let map = m2l.offset_map(d);
            assert_eq!(map.class, class_of_offset(o), "{o:?}");
            assert!(!m2l.has_offset(d));
            for i in 0..rings {
                for j in 0..n {
                    let ring = if map.flip_rings { rings - 1 - i } else { i };
                    let azimuth = if map.reverse {
                        (map.start + n - j) % n
                    } else {
                        (map.start + j) % n
                    };
                    assert_eq!(
                        (ring, azimuth),
                        symmetry.sample(rings, n, i, j),
                        "p = {p}, {o:?}, ({i}, {j})"
                    );
                }
            }
        }
    }
}

/// The trusted path sample by sample: the stored classes are ref's `Transfer` of the
/// representative times the factor, every offset's samples ref's permuted transfer function
/// times the factor bit for bit, and ref's transfer function of the offset itself within 4 ε.
fn check_samples(k: C) {
    let p = 6;
    let level = Level::new(k, 1.0);
    let grid = Grid::<f64>::new(p);
    let m2l = DiagonalM2l::<f64>::build(p, k, &ALL);
    let m2l32 = DiagonalM2l::<f32>::build(p, k, &ALL);
    let classes = classes();
    let mut transfers = Vec::new();
    for (c, class) in classes.iter().enumerate() {
        let rep = class.representative;
        let t = Transfer::new(&grid, &level, &frame([0; 3]), &frame(rep));
        let f = factor(&level, class.squared_distance);
        let expected: Vec<C> = t.samples().iter().map(|&s| s * f).collect();
        assert_eq!(
            bits(m2l.class_samples(c).unwrap()),
            bits(&expected),
            "{rep:?}"
        );
        let rounded: Vec<Complex<f32>> = expected
            .iter()
            .map(|w| Complex::new(w.re as f32, w.im as f32))
            .collect();
        assert_eq!(bits(m2l32.class_samples(c).unwrap()), bits(&rounded));
        transfers.push(t);
    }
    let mut w = vec![C::new(0.0, 0.0); m2l.samples()];
    for (d, o) in m2l_offsets().iter().enumerate() {
        let o = o.map(|x| x as i32);
        let (_, symmetry) = class_of(o);
        let c = class_of_offset(o);
        let f = factor(&level, classes[c].squared_distance);
        m2l.offset_samples(d, &mut w);
        let permuted: Vec<C> = transfers[c]
            .permuted(&grid, symmetry)
            .samples()
            .iter()
            .map(|&s| s * f)
            .collect();
        assert_eq!(bits(&w), bits(&permuted), "{o:?}");
        let direct = Transfer::new(&grid, &level, &frame([0; 3]), &frame(o));
        let largest = w.iter().fold(0.0f64, |m, s| m.max(s.norm()));
        let worst = w
            .iter()
            .zip(direct.samples())
            .fold(0.0f64, |m, (a, &b)| m.max((a - b * f).norm()));
        assert!(
            worst <= 4.0 * f64::EPSILON * largest,
            "{o:?}: {:.2} ε from ref's transfer function of the offset",
            worst / largest / f64::EPSILON
        );
    }
}

#[test]
fn the_samples_are_refs_transfer_functions() {
    check_samples(k_hat(5.0, 0.0));
    check_samples(k_hat(5.0, 0.1));
}

/// The pair against its naive loop over the offset's samples, every offset.
fn check_pair<T: RealScalar>() {
    let p = 5;
    let m2l = DiagonalM2l::<T>::build(p, k_hat(4.0, 0.0), &ALL);
    let n = m2l.samples();
    let to_t = |v: &[f64]| -> Vec<Complex<T>> {
        v.as_chunks::<2>()
            .0
            .iter()
            .map(|&[re, im]| Complex::new(T::from_f64(re), T::from_f64(im)))
            .collect()
    };
    let signature = to_t(&values(2 * n, 3));
    let start = to_t(&values(2 * n, 5));
    let mut w = vec![Complex::new(T::zero(), T::zero()); n];
    for d in 0..M2L_OFFSET_COUNT {
        let mut got = start.clone();
        m2l.pair(d, &signature, &mut got);
        m2l.offset_samples(d, &mut w);
        let expected: Vec<Complex<T>> = start
            .iter()
            .zip(&signature)
            .zip(&w)
            .map(|((&g, &f), &w)| g + w * f)
            .collect();
        assert_eq!(bits(&got), bits(&expected), "offset {d}");
    }
}

#[test]
fn the_pair_is_the_product_with_the_permuted_samples() {
    check_pair::<f64>();
    check_pair::<f32>();
}

#[test]
#[should_panic(expected = "which is not stored")]
fn a_pair_of_a_class_not_stored_panics() {
    let m2l = DiagonalM2l::<f64>::build(2, k_hat(3.0, 0.0), &[0, 5]);
    let d = m2l_offset_index([3, 3, 3]).unwrap();
    let s = vec![C::new(0.0, 0.0); m2l.samples()];
    let mut a = s.clone();
    m2l.pair(d, &s, &mut a);
}

#[test]
#[should_panic(expected = "ascend without repeats")]
fn unordered_classes_are_refused() {
    DiagonalM2l::<f64>::build(2, k_hat(3.0, 0.0), &[5, 0]);
}

#[test]
fn the_build_in_parts_the_bytes_and_the_bounds() {
    let (p, k) = (4, k_hat(6.0, 0.05));
    let stored = [0, 3, 7, 20, 33];
    let serial = DiagonalM2l::<f64>::build(p, k, &stored);
    let threads = DiagonalM2l::<f64>::build_on(p, k, &stored, &Threads);
    assert_eq!(serial.stored_classes(), stored);
    for c in 0..CLASS_COUNT {
        assert_eq!(serial.is_stored(c), stored.contains(&c));
        if let Some(s) = serial.class_samples(c) {
            assert_eq!(
                bits(s),
                bits(threads.class_samples(c).unwrap()),
                "class {c}"
            );
        } else {
            assert!(threads.class_samples(c).is_none());
        }
    }
    assert_eq!(
        serial.bytes(),
        DiagonalM2l::<f64>::bytes_for(p, stored.len())
    );
    let f32 = DiagonalM2l::<f32>::build(p, k, &stored);
    assert_eq!(f32.bytes(), DiagonalM2l::<f32>::bytes_for(p, stored.len()));
    assert_eq!((serial.degree(), serial.k_hat()), (p, k));

    let bounds = class_bounds(p, k, BoundConstants::F64);
    for (c, class) in classes().iter().enumerate() {
        let t = 2.0 * f64::from(class.squared_distance).sqrt();
        let e = nd_fmm_ref::helmholtz::diagonal::bound(k, t, 3f64.sqrt(), p, BoundConstants::F64);
        assert_eq!(bounds[c].to_bits(), e.to_bits(), "class {c}");
    }
    // The nearest classes bound the farthest.
    assert!(bounds[0] > bounds[CLASS_COUNT - 1]);
}

// ---------------------------------------------------------------------------------------
// The acceptance: against RCR on the same class within E(l, c).

/// Points in the box of half-width 1 at `centre`: `n` uniform and the eight corners.
fn in_box(centre: [f64; 3], n: usize, seed: u64) -> Vec<[f64; 3]> {
    let v = values(3 * n, seed);
    let mut points: Vec<[f64; 3]> = v
        .as_chunks::<3>()
        .0
        .iter()
        .map(|x| core::array::from_fn(|i| centre[i] + x[i]))
        .collect();
    for corner in 0..8 {
        let sign = |bit: usize| if corner >> bit & 1 == 1 { 1.0 } else { -1.0 };
        points.push(core::array::from_fn(|i| centre[i] + sign(i)));
    }
    points
}

fn charges(n: usize, seed: u64) -> Vec<C> {
    values(2 * n, seed)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[re, im]| C::new(re, im))
        .collect()
}

struct Field {
    potential: Vec<C>,
    gradient: Vec<[C; 3]>,
}

fn multipole(p: usize, level: &Level<f64>, centre: [f64; 3], seed: u64) -> Vec<f64> {
    let sources = in_box(centre, 40, seed);
    let q = charges(sources.len(), seed + 1);
    let mut out = vec![0.0; 2 * (p + 1) * (p + 1)];
    leaf::p2m(
        p,
        level,
        &Frame::new(centre, 1.0),
        &sources,
        &q,
        &mut Workspace::new(p),
        &mut out,
    );
    out
}

fn evaluate(
    p: usize,
    level: &Level<f64>,
    centre: [f64; 3],
    local: &[f64],
    x: &[[f64; 3]],
) -> Field {
    let mut field = Field {
        potential: vec![C::new(0.0, 0.0); x.len()],
        gradient: vec![[C::new(0.0, 0.0); 3]; x.len()],
    };
    leaf::l2p(
        p,
        level,
        &Frame::new(centre, 1.0),
        local,
        x,
        &mut Workspace::new(p),
        &mut field.potential,
        Some(&mut field.gradient),
    );
    field
}

fn rel_l2(a: impl Iterator<Item = C> + Clone, b: impl Iterator<Item = C>) -> f64 {
    let (mut diff, mut norm) = (0.0, 0.0);
    for (x, y) in a.zip(b) {
        diff += (x - y).norm_sqr();
        norm += y.norm_sqr();
    }
    (diff / norm).sqrt()
}

/// max(relative ℓ2 of φ̃, relative ℓ2 of ∇φ̃) of `a` against `b`.
fn error(a: &Field, b: &Field) -> f64 {
    let flat = |f: &Field| f.gradient.iter().flatten().copied().collect::<Vec<_>>();
    rel_l2(a.potential.iter().copied(), b.potential.iter().copied())
        .max(rel_l2(flat(a).into_iter(), flat(b).into_iter()))
}

/// The tables' level call for one pair: S → F, the pair, F → R into a zero local.
fn tables_local<T: RealScalar>(
    m2l: &DiagonalM2l<T>,
    conversions: &Conversions<T>,
    offset: usize,
    multipole: &[f64],
) -> Vec<f64> {
    let m: Vec<T> = multipole.iter().map(|&v| T::from_f64(v)).collect();
    let mut local = vec![T::zero(); m.len()];
    let mut buffers = LevelBuffers::new(conversions, 1, 1);
    m2l_level(
        conversions,
        &m,
        &[0],
        &mut local,
        &[0],
        &[0, 1],
        &[0],
        &mut buffers,
        |_, _, signature, accumulator| m2l.pair(offset, signature, accumulator),
    );
    local.iter().map(|&v| RealScalar::to_f64(v)).collect()
}

fn rcr_local(p: usize, level: &Level<f64>, o: [i32; 3], m: &[f64]) -> Vec<f64> {
    let (from, to) = (frame([0; 3]), frame(o));
    let table = rcr::m2l_table(p, p, level, &from, &to);
    let mut local = vec![0.0; m.len()];
    rcr::m2l(
        p,
        p,
        level,
        &from,
        &to,
        &table,
        &mut Workspace::new(p),
        m,
        &mut local,
    );
    local
}

/// ref's `direct::m2l`, the projection on a sphere (CH1.3c), a third path.
fn projected_local(p: usize, level: &Level<f64>, o: [i32; 3], m: &[f64]) -> Vec<f64> {
    let mut local = vec![0.0; m.len()];
    direct::m2l(
        p,
        p,
        level,
        &frame([0; 3]),
        &frame(o),
        &mut Workspace::new(p),
        m,
        &mut local,
    );
    local
}

/// Where the comparison with RCR is read at the floor of the bound.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Floor {
    /// Every offset against RCR within 2E: the acceptance.
    Strict,
    /// At large p, where f64 RCR rounds more than the diagonal form at E's floor: an offset
    /// above 2E against RCR passes if the diagonal form is no farther than max(2E, RCR's error)
    /// from a third path, ref's projection `direct::m2l`.
    AgainstTheProjection,
}

/// What the tables' M2L is measured against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reference {
    /// ref's RCR (the acceptance of CH4.2).
    Rcr,
    /// The same tables' M2L built directly (CH4.2c's "against the direct build").
    DirectBuild,
}

/// The tables' M2L of each offset in `offsets` against ref's RCR, within 2E of its class
/// (`floor` says how an offset above it is read), the tables built directly.
fn against_rcr<T: RealScalar>(
    ka: f64,
    eta: f64,
    p: usize,
    offsets: &[[i32; 3]],
    constants: BoundConstants,
    seed: u64,
    floor: Floor,
) {
    against::<T>(
        TransferBuild::Direct,
        Reference::Rcr,
        ka,
        eta,
        p,
        offsets,
        constants,
        seed,
        floor,
    );
}

/// The tables' M2L of each offset in `offsets`, the tables built by `build`, against
/// `reference`, within 2E of its class (`floor` says how an offset above it is read against
/// RCR; against the direct build every offset is read strictly).
#[allow(clippy::too_many_arguments)]
fn against<T: RealScalar>(
    build: TransferBuild,
    against: Reference,
    ka: f64,
    eta: f64,
    p: usize,
    offsets: &[[i32; 3]],
    constants: BoundConstants,
    seed: u64,
    floor: Floor,
) {
    let k = k_hat(ka, eta);
    let level = Level::new(k, 1.0);
    let needed: Vec<usize> = {
        let mut c: Vec<usize> = offsets.iter().map(|&o| class_of_offset(o)).collect();
        c.sort_unstable();
        c.dedup();
        c
    };
    let m2l = DiagonalM2l::<T>::build_by(p, k, &needed, build, &Serial);
    let direct =
        (against == Reference::DirectBuild).then(|| DiagonalM2l::<T>::build(p, k, &needed));
    let conversions = Conversions::<T>::new(p, level.scale());
    let bounds = class_bounds(p, k, constants);
    let mut worst: f64 = 0.0;
    let mut failures = Vec::new();
    for (i, &o) in offsets.iter().enumerate() {
        let d = m2l_offset_index(o.map(i64::from)).unwrap();
        let e = bounds[m2l.offset_map(d).class];
        let m = multipole(p, &level, [0.0; 3], seed + 7 * i as u64);
        let targets = in_box(frame(o).centre, 24, seed + 7 * i as u64 + 3);
        let centre = frame(o).centre;
        let reference_local = match &direct {
            Some(direct) => tables_local(direct, &conversions, d, &m),
            None => rcr_local(p, &level, o, &m),
        };
        let reference = evaluate(p, &level, centre, &reference_local, &targets);
        let got = evaluate(
            p,
            &level,
            centre,
            &tables_local(&m2l, &conversions, d, &m),
            &targets,
        );
        let measured = error(&got, &reference);
        assert!(measured.is_finite(), "{o:?}: not finite");
        worst = worst.max(measured / e);
        if std::env::var_os("DIAGONAL_DETAIL").is_some() {
            println!("  {o:?}: measured {measured:.3e}, E {e:.3e}");
        }
        if measured <= e / THETA {
            continue;
        }
        if floor == Floor::Strict || against == Reference::DirectBuild {
            failures.push(format!("{o:?}: {measured:.2e} > 2E = {:.2e}", e / THETA));
            continue;
        }
        let projected = evaluate(
            p,
            &level,
            centre,
            &projected_local(p, &level, o, &m),
            &targets,
        );
        let (diagonal, rcr) = (error(&got, &projected), error(&reference, &projected));
        println!(
            "  {o:?} at the floor: against RCR {measured:.2e} > 2E = {:.2e}; against the \
             projection the diagonal form {diagonal:.2e}, RCR {rcr:.2e}",
            e / THETA
        );
        if diagonal > (e / THETA).max(rcr) {
            failures.push(format!(
                "{o:?}: {diagonal:.2e} from the projection > max(2E, RCR's {rcr:.2e})"
            ));
        }
    }
    println!(
        "{build:?} against {against:?}: k a = {ka}, η = {eta}, p = {p}, {} offsets: largest \
         measured/E {worst:.3}",
        offsets.len()
    );
    assert!(failures.is_empty(), "k a = {ka}, p = {p}: {failures:?}");
}

fn representatives() -> Vec<[i32; 3]> {
    classes().iter().map(|c| c.representative).collect()
}

/// Acceptance (proposal §10 CH4.2): the tables' diagonal M2L against RCR on the 34 classes
/// within E(l, c) (measured ≤ 2E), for real k at k a = 2.83 (p = 7, S6's worst ratio) and 8,
/// weakly damped k at k a = 8 (Im k = Re k / 10), p from §3.5's formula at 10⁻⁶.
#[test]
fn against_rcr_on_the_classes_within_the_bound() {
    let reps = representatives();
    let cases = [
        (2.83, 0.0, 7),
        (8.0, 0.0, formula_degree(8.0, 1e-6)),
        (8.0, 0.1, formula_degree(8.0, 1e-6)),
    ];
    for (i, (ka, eta, p)) in cases.into_iter().enumerate() {
        against_rcr::<f64>(
            ka,
            eta,
            p,
            &reps,
            BoundConstants::F64,
            100 * i as u64,
            Floor::Strict,
        );
    }
}

/// The 316 offsets through their classes' maps against RCR within 2E, at k a = 5, p = 16.
#[test]
fn every_offset_against_rcr_within_the_bound() {
    let offsets: Vec<[i32; 3]> = m2l_offsets().iter().map(|o| o.map(|x| x as i32)).collect();
    against_rcr::<f64>(
        5.0,
        0.0,
        16,
        &offsets,
        BoundConstants::F64,
        7000,
        Floor::Strict,
    );
}

/// f32 (the tables rounded from f64, the conversions in f32) against RCR in f64 within the
/// f32 bound (C_T = 0.18, C₀ = 50, u = 2⁻²⁴), within 2E: k a = 8, p = 16.
#[test]
fn f32_against_rcr_within_its_bound() {
    let reps = representatives();
    against_rcr::<f32>(
        8.0,
        0.0,
        16,
        &reps,
        BoundConstants::F32,
        9000,
        Floor::Strict,
    );
}

/// The acceptance's larger cases, within 2E: f64 at k a = 16 (the formula's p at 10⁻⁶), f32 at
/// k a ≈ 16, p = 30 (D3's range). Release only (about 10 s).
#[test]
#[ignore = "larger degrees: run in release by hand"]
fn against_rcr_at_k_a_16() {
    let reps = representatives();
    let p = formula_degree(16.0, 1e-6);
    against_rcr::<f64>(16.0, 0.0, p, &reps, BoundConstants::F64, 300, Floor::Strict);
    against_rcr::<f32>(
        16.0,
        0.0,
        30,
        &reps,
        BoundConstants::F32,
        9100,
        Floor::Strict,
    );
}

/// The acceptance at large degrees: k a = 32 (the formula at 10⁻⁸), 54 (p = 72, level 3 of W1
/// at kD = 500 calibrated, and 82, the formula at 10⁻⁶) on the 34 classes, and 108 (p = 131,
/// level 2) on the nearest, a middle and the farthest class. Here E is near its floor C₀ u
/// on the far classes and f64 RCR rounds more than the diagonal form: an offset above 2E
/// against RCR is read against the projection ([`Floor::AgainstTheProjection`]). Measured on
/// the M6: 2.4 E against RCR on (2, 2, 2) at k a = 54, p = 82, and 3.7 E on (3, 3, 3) at
/// k a = 108; the diagonal form nearer the projection than RCR on both
/// (docs/helmholtz/tasks/CH4.2a.md). Release only (about 2 minutes).
#[test]
#[ignore = "large degrees: run in release by hand"]
fn against_rcr_at_large_degrees() {
    let reps = representatives();
    let floor = Floor::AgainstTheProjection;
    for (i, (ka, p)) in [(32.0, formula_degree(32.0, 1e-8)), (54.0, 72), (54.0, 82)]
        .into_iter()
        .enumerate()
    {
        against_rcr::<f64>(
            ka,
            0.0,
            p,
            &reps,
            BoundConstants::F64,
            20_000 + i as u64,
            floor,
        );
    }
    let few = [[0, 0, 2], [1, 2, 2], [3, 3, 3]];
    against_rcr::<f64>(108.0, 0.0, 131, &few, BoundConstants::F64, 21_000, floor);
}

// ---------------------------------------------------------------------------------------
// The interpolated build (CH4.2c1): against the direct build and RCR within 2E at C_T = 0.23.

const INTERPOLATED: BoundConstants = BoundConstants::F64_INTERPOLATED;

/// Every stored sample of the interpolated build against the direct build's: the largest
/// difference per class over the class's largest sample, the largest over the classes. Both in
/// f64.
fn interpolated_against_direct(p: usize, k: C) -> f64 {
    let direct = DiagonalM2l::<f64>::build(p, k, &ALL);
    let interpolated =
        DiagonalM2l::<f64>::build_by(p, k, &ALL, TransferBuild::Interpolated, &Serial);
    (0..CLASS_COUNT)
        .map(|c| {
            let (a, b) = (
                interpolated.class_samples(c).unwrap(),
                direct.class_samples(c).unwrap(),
            );
            let largest = b.iter().fold(0.0f64, |m, s| m.max(s.norm()));
            let worst = a
                .iter()
                .zip(b)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()));
            assert!(worst.is_finite(), "class {c}");
            worst / largest
        })
        .fold(0.0, f64::max)
}

/// Test 1 of the plan: every sample of every class, the interpolated build against the direct
/// build within 64 u (32 ε) of the class's largest sample, at p = 6 (k a = 5, real and damped)
/// and p = 16 (k a = 5 and 8).
#[test]
fn interpolated_samples_against_the_direct_build() {
    for (p, ka, eta) in [(6, 5.0, 0.0), (6, 5.0, 0.1), (16, 5.0, 0.0), (16, 8.0, 0.1)] {
        let worst = interpolated_against_direct(p, k_hat(ka, eta));
        println!(
            "p = {p}, k a = {ka}, η = {eta}: largest difference {:.2} ε of the class's largest",
            worst / f64::EPSILON
        );
        assert!(
            worst <= 32.0 * f64::EPSILON,
            "p = {p}, k a = {ka}: {:.1} ε",
            worst / f64::EPSILON
        );
    }
}

/// Test 2: the acceptance with the interpolated tables, against RCR within 2E at C_T = 0.23 on
/// the 34 classes (k a = 2.83, p = 7; k a = 8 real and damped), the 316 offsets (k a = 5,
/// p = 16), and f32 within `F32`'s 2E (k a = 8, p = 16).
#[test]
fn interpolated_against_rcr_within_the_bound() {
    let reps = representatives();
    let offsets: Vec<[i32; 3]> = m2l_offsets().iter().map(|o| o.map(|x| x as i32)).collect();
    let build = TransferBuild::Interpolated;
    assert_eq!(build.bound_constants(false), INTERPOLATED);
    assert_eq!(build.bound_constants(true), BoundConstants::F32);
    let cases: [(f64, f64, usize, &[[i32; 3]]); 4] = [
        (2.83, 0.0, 7, &reps),
        (8.0, 0.0, formula_degree(8.0, 1e-6), &reps),
        (8.0, 0.1, formula_degree(8.0, 1e-6), &reps),
        (5.0, 0.0, 16, &offsets),
    ];
    for (i, (ka, eta, p, set)) in cases.into_iter().enumerate() {
        against::<f64>(
            build,
            Reference::Rcr,
            ka,
            eta,
            p,
            set,
            INTERPOLATED,
            30_000 + 100 * i as u64,
            Floor::Strict,
        );
    }
    against::<f32>(
        build,
        Reference::Rcr,
        8.0,
        0.0,
        16,
        &reps,
        build.bound_constants(true),
        31_000,
        Floor::Strict,
    );
}

/// Test 3: the interpolated tables' M2L against the direct tables' within 2E at C_T = 0.23, on
/// the cases of test 2.
#[test]
fn interpolated_against_the_direct_build_within_the_bound() {
    let reps = representatives();
    let offsets: Vec<[i32; 3]> = m2l_offsets().iter().map(|o| o.map(|x| x as i32)).collect();
    let cases: [(f64, f64, usize, &[[i32; 3]]); 4] = [
        (2.83, 0.0, 7, &reps),
        (8.0, 0.0, formula_degree(8.0, 1e-6), &reps),
        (8.0, 0.1, formula_degree(8.0, 1e-6), &reps),
        (5.0, 0.0, 16, &offsets),
    ];
    for (i, (ka, eta, p, set)) in cases.into_iter().enumerate() {
        against::<f64>(
            TransferBuild::Interpolated,
            Reference::DirectBuild,
            ka,
            eta,
            p,
            set,
            INTERPOLATED,
            32_000 + 100 * i as u64,
            Floor::Strict,
        );
    }
}

/// Test 4: both builds in parts of one ring on a map that runs them in reverse on three
/// threads, bit for bit the serial build; the direct build by `build_by` bit for bit `build`
/// (CH4.2a's tables, checked against ref's `Transfer` above); f32 the f64 tables rounded.
#[test]
fn both_builds_in_parts_by_ring() {
    let (p, k) = (5, k_hat(6.0, 0.05));
    let stored = [0, 3, 7, 20, 33];
    for build in [TransferBuild::Direct, TransferBuild::Interpolated] {
        let serial = DiagonalM2l::<f64>::build_by(p, k, &stored, build, &Serial);
        let threads = DiagonalM2l::<f64>::build_by(p, k, &stored, build, &Threads);
        let single = DiagonalM2l::<f32>::build_by(p, k, &stored, build, &Threads);
        for &c in &stored {
            let s = serial.class_samples(c).unwrap();
            assert_eq!(
                bits(s),
                bits(threads.class_samples(c).unwrap()),
                "{build:?}, class {c}"
            );
            let rounded: Vec<Complex<f32>> = s
                .iter()
                .map(|w| Complex::new(w.re as f32, w.im as f32))
                .collect();
            assert_eq!(bits(single.class_samples(c).unwrap()), bits(&rounded));
        }
        assert_eq!(serial.stored_classes(), stored);
    }
    let by_class = DiagonalM2l::<f64>::build(p, k, &stored);
    let by_ring = DiagonalM2l::<f64>::build_by(p, k, &stored, TransferBuild::Direct, &Threads);
    for &c in &stored {
        assert_eq!(
            bits(by_class.class_samples(c).unwrap()),
            bits(by_ring.class_samples(c).unwrap())
        );
    }
    assert_eq!(interpolation_steps(p), 6 * p + 1);
}

/// The interpolated acceptance's larger cases (release only): against RCR within 2E at
/// C_T = 0.23 at k a = 16 (the formula's p at 10⁻⁶) and at the large degrees of
/// `against_rcr_at_large_degrees` (k a 32–108, read against the projection at E's floor), and
/// against the direct build at k a = 16 and 54 (p = 72).
#[test]
#[ignore = "larger degrees: run in release by hand"]
fn interpolated_at_large_degrees() {
    let reps = representatives();
    let build = TransferBuild::Interpolated;
    let p16 = formula_degree(16.0, 1e-6);
    against::<f64>(
        build,
        Reference::Rcr,
        16.0,
        0.0,
        p16,
        &reps,
        INTERPOLATED,
        33_000,
        Floor::Strict,
    );
    for (i, (ka, p)) in [(16.0, p16), (54.0, 72)].into_iter().enumerate() {
        against::<f64>(
            build,
            Reference::DirectBuild,
            ka,
            0.0,
            p,
            &reps,
            INTERPOLATED,
            33_100 + i as u64,
            Floor::Strict,
        );
    }
    let floor = Floor::AgainstTheProjection;
    for (i, (ka, p)) in [(32.0, formula_degree(32.0, 1e-8)), (54.0, 72), (54.0, 82)]
        .into_iter()
        .enumerate()
    {
        against::<f64>(
            build,
            Reference::Rcr,
            ka,
            0.0,
            p,
            &reps,
            INTERPOLATED,
            34_000 + i as u64,
            floor,
        );
    }
    let few = [[0, 0, 2], [1, 2, 2], [3, 3, 3]];
    against::<f64>(
        build,
        Reference::Rcr,
        108.0,
        0.0,
        131,
        &few,
        INTERPOLATED,
        35_000,
        floor,
    );
}

/// Test 6 (prints, asserts no time): one class's build on one thread, direct against
/// interpolated, at the degrees of W1's diagonal levels (p = 41, 72, 131 and 245 at
/// k a ≈ 27, 54, 108 and 216), with the samples' largest difference. Release only.
#[test]
#[ignore = "timing: run in release by hand"]
fn time_one_class() {
    let classes = [classes()
        .iter()
        .position(|c| c.representative == [0, 0, 2])
        .unwrap()];
    println!("| p | k a | direct s | interpolated s | ratio | largest difference (ε) |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: |");
    for (p, ka) in [(41, 27.0), (72, 54.0), (131, 108.0), (245, 216.0)] {
        let k = k_hat(ka, 0.0);
        let clock = std::time::Instant::now();
        let direct = DiagonalM2l::<f64>::build(p, k, &classes);
        let t_direct = clock.elapsed().as_secs_f64();
        let clock = std::time::Instant::now();
        let interpolated =
            DiagonalM2l::<f64>::build_by(p, k, &classes, TransferBuild::Interpolated, &Serial);
        let t_interpolated = clock.elapsed().as_secs_f64();
        let (a, b) = (
            interpolated.class_samples(classes[0]).unwrap(),
            direct.class_samples(classes[0]).unwrap(),
        );
        let largest = b.iter().fold(0.0f64, |m, s| m.max(s.norm()));
        let worst = a
            .iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()));
        println!(
            "| {p} | {ka} | {t_direct:.3} | {t_interpolated:.3} | {:.1} | {:.2} |",
            t_direct / t_interpolated,
            worst / largest / f64::EPSILON
        );
    }
}

// ---------------------------------------------------------------------------------------
// Determinism with the RCR pairs added after F → R.

/// A level of boxes of half-width 1 (centres 2 apart in each direction) on a 6 × 6 × 2 block:
/// the sources every box in reverse order, five targets each with its V row in the sources'
/// order, split into the diagonal pairs (classes with |d|² ≥ 9) and the RCR pairs (the rest).
struct Mixed {
    cells: Vec<[i32; 3]>,
    sources: Vec<usize>,
    targets: Vec<usize>,
    /// Per target its diagonal pairs: the source slots and the offsets' table indices.
    rows: Vec<usize>,
    slots: Vec<usize>,
    offsets: Vec<usize>,
    /// Per target its RCR pairs, (source box, offset index), in row order.
    rcr: Vec<Vec<(usize, usize)>>,
    diagonal_classes: Vec<usize>,
}

fn mixed() -> Mixed {
    let cells: Vec<[i32; 3]> = (0..2)
        .flat_map(|z| (0..6).flat_map(move |y| (0..6).map(move |x| [x, y, z])))
        .collect();
    let sources: Vec<usize> = (0..cells.len()).rev().collect();
    let targets = vec![14, 0, 35, 50, 21];
    let classes = classes();
    let (mut rows, mut slots, mut offsets, mut rcr) = (vec![0], Vec::new(), Vec::new(), Vec::new());
    let mut diagonal_classes = Vec::new();
    for &t in &targets {
        let mut row_rcr = Vec::new();
        for (slot, &s) in sources.iter().enumerate() {
            let o: [i32; 3] = core::array::from_fn(|a| cells[t][a] - cells[s][a]);
            let Some(d) = m2l_offset_index(o.map(i64::from)) else {
                continue;
            };
            let c = class_of_offset(o);
            if classes[c].squared_distance >= 9 {
                slots.push(slot);
                offsets.push(d);
                diagonal_classes.push(c);
            } else {
                row_rcr.push((s, d));
            }
        }
        rows.push(slots.len());
        rcr.push(row_rcr);
    }
    diagonal_classes.sort_unstable();
    diagonal_classes.dedup();
    Mixed {
        cells,
        sources,
        targets,
        rows,
        slots,
        offsets,
        rcr,
        diagonal_classes,
    }
}

/// The mixed level: the diagonal pairs through `m2l_level` with chunks of `chunk`, then each
/// target's RCR pairs through `HelmholtzTables`' M2L in row order (`None`: every pair RCR, in
/// the same rows). Returns the locals.
fn mixed_level(
    c: &Mixed,
    m2l: &DiagonalM2l<f64>,
    rcr_tables: &HelmholtzTables<f64>,
    multipoles: &[f64],
    start: &[f64],
    chunk: Option<usize>,
) -> Vec<f64> {
    let p = m2l.degree();
    let len = 2 * (p + 1) * (p + 1);
    let mut locals = start.to_vec();
    let mut scratch = rcr_tables.scratch();
    let mut rcr_pair = |s: usize, d: usize, t: usize, locals: &mut [f64]| {
        rcr_tables.m2l(
            2,
            d,
            &multipoles[s * len..(s + 1) * len],
            2,
            &mut locals[t * len..(t + 1) * len],
            1,
            &mut scratch,
        );
    };
    match chunk {
        Some(chunk) => {
            let level = Level::new(m2l.k_hat(), 1.0);
            let conversions = Conversions::<f64>::new(p, level.scale());
            let mut buffers = LevelBuffers::new(&conversions, c.sources.len(), chunk);
            m2l_level(
                &conversions,
                multipoles,
                &c.sources,
                &mut locals,
                &c.targets,
                &c.rows,
                &c.slots,
                &mut buffers,
                |_, k, signature, accumulator| m2l.pair(c.offsets[k], signature, accumulator),
            );
            for (i, &t) in c.targets.iter().enumerate() {
                for &(s, d) in &c.rcr[i] {
                    rcr_pair(s, d, t, &mut locals);
                }
            }
        }
        None => {
            for (i, &t) in c.targets.iter().enumerate() {
                for k in c.rows[i]..c.rows[i + 1] {
                    rcr_pair(c.sources[c.slots[k]], c.offsets[k], t, &mut locals);
                }
                for &(s, d) in &c.rcr[i] {
                    rcr_pair(s, d, t, &mut locals);
                }
            }
        }
    }
    locals
}

/// The level with the RCR pairs added after F → R: bit for bit at every chunk size, and
/// against the level with every pair RCR within 2 max E of the diagonal classes used, in the
/// measure of the acceptance over every target box.
#[test]
fn the_rcr_pairs_after_f2r_at_every_chunk_size() {
    let c = mixed();
    assert!(c.rows.windows(2).all(|w| w[1] > w[0]));
    assert!(c.rcr.iter().all(|r| !r.is_empty()));
    let (ka, p) = (8.0, 12);
    let k = k_hat(ka, 0.0);
    let level = Level::new(k, 1.0);
    let m2l = DiagonalM2l::<f64>::build(p, k, &c.diagonal_classes);
    let rcr_tables = HelmholtzTables::<f64>::build(k * 4.0, &[p, p, p], DEFAULT_BLOCK_BUDGET);
    assert_eq!(rcr_tables.k_hat(2), k);
    let centre = |b: usize| c.cells[b].map(|x| 2.0 * f64::from(x));
    let multipoles: Vec<f64> = (0..c.cells.len())
        .flat_map(|b| multipole(p, &level, centre(b), 300 + b as u64))
        .collect();
    let len = 2 * (p + 1) * (p + 1);
    let zero = vec![0.0; c.cells.len() * len];

    let whole = mixed_level(
        &c,
        &m2l,
        &rcr_tables,
        &multipoles,
        &zero,
        Some(c.targets.len()),
    );
    for chunk in [1, 2, 3, 4, 5, 7] {
        assert_eq!(
            real_bits(&mixed_level(
                &c,
                &m2l,
                &rcr_tables,
                &multipoles,
                &zero,
                Some(chunk)
            )),
            real_bits(&whole),
            "chunks of {chunk}"
        );
    }
    // From a nonzero start too (the locals of the level's other pairs).
    let start = values(c.cells.len() * len, 41);
    let from_start = mixed_level(&c, &m2l, &rcr_tables, &multipoles, &start, Some(1));
    assert_eq!(
        real_bits(&from_start),
        real_bits(&mixed_level(
            &c,
            &m2l,
            &rcr_tables,
            &multipoles,
            &start,
            Some(5)
        ))
    );

    let all_rcr = mixed_level(&c, &m2l, &rcr_tables, &multipoles, &zero, None);
    let bounds = class_bounds(p, k, BoundConstants::F64);
    let e = c
        .diagonal_classes
        .iter()
        .map(|&cl| bounds[cl])
        .fold(0.0f64, f64::max);
    let mut worst: f64 = 0.0;
    for &t in &c.targets {
        let x = in_box(centre(t), 24, 500 + t as u64);
        let local = |l: &[f64]| evaluate(p, &level, centre(t), &l[t * len..(t + 1) * len], &x);
        worst = worst.max(error(&local(&whole), &local(&all_rcr)));
    }
    println!("mixed level: measured {worst:.3e}, max E {e:.3e}");
    assert!(worst <= e / THETA, "{worst:.2e} > 2E = {:.2e}", e / THETA);
}

// ---------------------------------------------------------------------------------------
// CH4.3a: the bound on the FMM's levels (docs/helmholtz/tasks/CH4.3a.md).

/// The calibrated degree of `nd_fmm_exec::helmholtz::DegreeRule::CALIBRATED` (proposal §3.5,
/// CH3.3), restated: this crate cannot depend on nd-fmm-exec. x = Re(k) a, L = log₁₀(1/ε); the
/// ceilings forgive 10⁻⁹ as the rule's do.
fn calibrated_degree(x: f64, eps: f64) -> usize {
    let l = (1.0 / eps).log10();
    let ceil = |v: f64| (v - 1e-9).ceil() as usize;
    let plateau = ceil(0.30 * l * l + 0.25 * l + 2.60);
    let high = ceil(x + 0.80 * (1.0 / eps).ln().powf(2.0 / 3.0) * x.cbrt());
    plateau.max(high)
}

/// The field of complex charges `q` at `sources` on the targets `x`, Σ q e^{ikr}/r and its
/// gradient in x, in the units of `leaf::l2p` on a level of half-width 1 (1/(4π) left out,
/// §3.1).
fn direct_field(k: C, sources: &[[f64; 3]], q: &[C], x: &[[f64; 3]]) -> Field {
    let i = C::new(0.0, 1.0);
    let mut field = Field {
        potential: vec![C::new(0.0, 0.0); x.len()],
        gradient: vec![[C::new(0.0, 0.0); 3]; x.len()],
    };
    for (t, xt) in x.iter().enumerate() {
        for (y, &qj) in sources.iter().zip(q) {
            let d: [f64; 3] = core::array::from_fn(|c| xt[c] - y[c]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let e = (i * k * r).exp() * qj;
            field.potential[t] += e / r;
            let radial = e * (i * k * r - 1.0) / (r * r * r);
            for (g, &dc) in field.gradient[t].iter_mut().zip(&d) {
                *g += radial * dc;
            }
        }
    }
    field
}

/// max(‖φ̃_a − φ̃_b‖ / ‖φ̃_s‖, ‖∇φ̃_a − ∇φ̃_b‖ / ‖∇φ̃_s‖): the difference of `a` and `b` relative
/// to the field `scale`.
fn error_over(a: &Field, b: &Field, scale: &Field) -> f64 {
    let flat = |f: &Field| f.gradient.iter().flatten().copied().collect::<Vec<_>>();
    let norm = |v: &[C]| v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let diff = |u: &[C], v: &[C]| {
        u.iter()
            .zip(v)
            .map(|(x, y)| (x - y).norm_sqr())
            .sum::<f64>()
    };
    let phi = diff(&a.potential, &b.potential).sqrt() / norm(&scale.potential);
    let (ga, gb, gs) = (flat(a), flat(b), flat(scale));
    phi.max(diff(&ga, &gb).sqrt() / norm(&gs))
}

/// One (build, k a, η, ε) of the study: the 34 classes at the calibrated degree.
struct StudyRow {
    build: TransferBuild,
    ka: f64,
    eta: f64,
    eps: f64,
    p: usize,
    /// The largest per-pair relative error over E (CH1.3d1's measure, `against`'s), with the
    /// classes read against the projection at E's floor counted at their projection's error.
    worst: f64,
    /// The largest input-relative error (over the undamped field) over E, and over E without
    /// the damping factor e^{Im(k)(|t| + 2a)}.
    input_worst: f64,
    input_worst_without: f64,
    /// The classes the rule takes (E ≤ θ ε), with and without the damping factor.
    passes: usize,
    passes_without: usize,
    /// The classes the input-relative measure keeps within ε, (the measured error ≤ ε).
    input_within: usize,
    /// The real-k unit check: the RCR chain against the direct sum (relative, η = 0 only).
    units: f64,
    failures: Vec<String>,
}

fn study_row(build: TransferBuild, ka: f64, eta: f64, eps: f64, seed: u64) -> StudyRow {
    let p = calibrated_degree(ka, eps);
    let k = k_hat(ka, eta);
    let level = Level::new(k, 1.0);
    let constants = build.bound_constants(false);
    let m2l = DiagonalM2l::<f64>::build_by(p, k, &ALL, build, &Serial);
    let conversions = Conversions::<f64>::new(p, level.scale());
    let bounds = class_bounds(p, k, constants);
    let a = 3f64.sqrt();
    let floor = constants.c_0 * constants.unit_roundoff;
    let mut row = StudyRow {
        build,
        ka,
        eta,
        eps,
        p,
        worst: 0.0,
        input_worst: 0.0,
        input_worst_without: 0.0,
        passes: 0,
        passes_without: 0,
        input_within: 0,
        units: 0.0,
        failures: Vec::new(),
    };
    for (c, class) in classes().iter().enumerate() {
        let o = class.representative;
        let e = bounds[c];
        let t = 2.0 * f64::from(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        let e_without = (e - floor) * (-(k.im * (t + 2.0 * a))).exp() + floor;
        row.passes += usize::from(e <= THETA * eps);
        row.passes_without += usize::from(e_without <= THETA * eps);
        let s = seed + 7 * c as u64;
        let sources = in_box([0.0; 3], 40, s);
        let q = charges(sources.len(), s + 1);
        let m = multipole(p, &level, [0.0; 3], s);
        let centre = frame(o).centre;
        let targets = in_box(centre, 24, s + 3);
        let d = m2l_offset_index(o.map(i64::from)).unwrap();
        let reference = evaluate(p, &level, centre, &rcr_local(p, &level, o, &m), &targets);
        let got = evaluate(
            p,
            &level,
            centre,
            &tables_local(&m2l, &conversions, d, &m),
            &targets,
        );
        let mut measured = error(&got, &reference);
        if !measured.is_finite() {
            row.failures.push(format!("{o:?}: not finite"));
            continue;
        }
        if measured > e / THETA {
            // E's floor at large p: read against ref's projection, as `against` does.
            let projected = evaluate(
                p,
                &level,
                centre,
                &projected_local(p, &level, o, &m),
                &targets,
            );
            let (diagonal, rcr) = (error(&got, &projected), error(&reference, &projected));
            if p >= 72 && diagonal <= (e / THETA).max(rcr) {
                measured = diagonal.min(measured);
            } else {
                row.failures.push(format!(
                    "{o:?}: {measured:.2e} > 2E = {:.2e} (projection: {diagonal:.2e}, RCR \
                     {rcr:.2e})",
                    e / THETA
                ));
            }
        }
        row.worst = row.worst.max(measured / e);
        let undamped = direct_field(C::new(k.re, 0.0), &sources, &q, &targets);
        let input = error_over(&got, &reference, &undamped);
        row.input_worst = row.input_worst.max(input / e);
        row.input_worst_without = row.input_worst_without.max(input / e_without);
        row.input_within += usize::from(input <= eps);
        if eta == 0.0 {
            row.units = row.units.max(error_over(&reference, &undamped, &undamped));
        }
    }
    row
}

/// CH4.3a's pair evidence (docs/helmholtz/tasks/CH4.3a.md): the 34 classes at the (k a, p) of
/// the FMM's levels on W1 and W3 at kD = 100, 200, 500 and 1000 (k a = kD √3 / 2^{l+1} for the
/// levels l ≥ 2 down to k a ≈ 1.35; 216 at kD = 1000's level 2 is CH4.3c's), the calibrated p at
/// ε = 10⁻³, 10⁻⁴, 10⁻⁶ and 10⁻⁸, real k and η = Im k / Re k = 0.1 (every k a: W1 at
/// kD = 500 + 50i) and 1 (k a ≤ 21.65, kD = 100 + 100i's levels; at k a = 108 and η = 1 the
/// transfer functions leave f64's range), both builds of the transfer functions. Asserted, in S6's
/// calibrated range k a ≥ 2.8 (`S6_MIN_KA`): every class within 2E per pair against RCR (read
/// against ref's projection at E's floor, p ≥ 72); everywhere: the RCR chain against the direct
/// sum at real k within 10⁻⁴ at 10⁻⁸ (the units of the input-relative measure). Below k a = 2.8 a
/// class above 2E is printed as a finding (on the M6: the direct build's (0, 0, 3) at k a = 1.69,
/// η = 0.1, 10⁻⁴, 2.9 E with E ≪ ε; CH4.3a's record). Printed per row: the largest measured over E
/// per pair and input-relative (over the undamped field, CH1.3c's input-side weighting), also over
/// E without the damping factor, and the classes each would take. Release, one case per thread on
/// the machine's threads; about nine minutes on the M6.
/// The smallest k a of S6's sweep (2.83, `spikes/diagonal/SPIKE_REPORT.md` §2), on which C_T
/// was fitted.
const S6_MIN_KA: f64 = 2.8;

#[test]
#[ignore = "a study on the FMM's levels: run in release by hand"]
fn the_bound_on_the_fmms_levels() {
    let mut kas: Vec<f64> = Vec::new();
    for kd in [100.0, 200.0, 500.0, 1000.0f64] {
        let mut l = 2;
        loop {
            let ka = kd * 3f64.sqrt() / f64::from(1u32 << (l + 1));
            if ka < 1.3 {
                break;
            }
            if ka < 200.0 && !kas.iter().any(|&x| (x - ka).abs() < 1e-9) {
                kas.push(ka);
            }
            l += 1;
        }
    }
    kas.sort_by(|a, b| b.total_cmp(a));
    let mut cases = Vec::new();
    for &eta in &[0.0, 0.1, 1.0] {
        for &ka in kas.iter().filter(|&&ka| eta < 1.0 || ka < 22.0) {
            for &eps in &[1e-3, 1e-4, 1e-6, 1e-8] {
                for build in [TransferBuild::Direct, TransferBuild::Interpolated] {
                    cases.push((build, ka, eta, eps));
                }
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(build, ka, eta, eps)) = cases.get(i) else {
                        break;
                    };
                    let row = study_row(build, ka, eta, eps, 40_000 + 1_000 * i as u64);
                    rows.lock().unwrap().push((i, row));
                }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|(i, _)| *i);
    println!(
        "| build | k a | η | ε | p | per pair ÷ E | input ÷ E | input ÷ E w/o factor | rule \
         | rule w/o factor | input ≤ ε | units |"
    );
    println!(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    let mut failures = Vec::new();
    for (_, r) in &rows {
        println!(
            "| {:?} | {:.2} | {} | {:.0e} | {} | {:.3} | {:.2e} | {:.2e} | {}/34 | {}/34 | {}/34 \
             | {} |",
            r.build,
            r.ka,
            r.eta,
            r.eps,
            r.p,
            r.worst,
            r.input_worst,
            r.input_worst_without,
            r.passes,
            r.passes_without,
            r.input_within,
            if r.eta == 0.0 {
                format!("{:.1e}", r.units)
            } else {
                "–".into()
            },
        );
        for f in &r.failures {
            if r.ka < S6_MIN_KA {
                println!(
                    "FINDING below S6's range: {:?} k a {:.2} η {} ε {:e}: {f}",
                    r.build, r.ka, r.eta, r.eps
                );
                continue;
            }
            failures.push(format!(
                "{:?} k a {:.2} η {} ε {:e}: {f}",
                r.build, r.ka, r.eta, r.eps
            ));
        }
        if r.eta == 0.0 && r.eps == 1e-8 && r.units > 1e-4 {
            failures.push(format!("units at k a {:.2}: {:.2e}", r.ka, r.units));
        }
    }
    for f in &failures {
        println!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} failures", failures.len());
}

//! A level's transfer functions of the diagonal form and the pair (Helmholtz CH4.2a,
//! `nd_fmm_tables::helmholtz::diagonal::transfer`; proposal §3.6, §5.4;
//! `docs/helmholtz/tasks/CH4.2a.md`).
//!
//! Error measures:
//! - the maps, the stored samples against ref's `Transfer` times the factor, the permuted
//!   samples, the pair against its naive loop, the build in parts, the level at every chunk
//!   size: bit for bit (`to_bits`);
//! - an offset's samples against ref's `Transfer` of the offset itself: the largest absolute
//!   difference within 4 u of the largest sample;
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
    CLASS_COUNT, Conversions, DiagonalM2l, LevelBuffers, class_bounds, m2l_level,
};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};

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
/// times the factor bit for bit, and ref's transfer function of the offset itself within 4 u.
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
            "{o:?}: {:.2} u from ref's transfer function of the offset",
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
        v.chunks_exact(2)
            .map(|z| Complex::new(T::from_f64(z[0]), T::from_f64(z[1])))
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
        .chunks_exact(3)
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
        .chunks_exact(2)
        .map(|z| C::new(z[0], z[1]))
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

/// The tables' M2L of each offset in `offsets` against ref's RCR, within 2E of its class, or
/// where it is not (E's floor at large p, where f64 RCR rounds more than the diagonal form),
/// within 2E plus RCR's own error against the projection; returns the largest measured/E.
fn against_rcr<T: RealScalar>(
    ka: f64,
    eta: f64,
    p: usize,
    offsets: &[[i32; 3]],
    constants: BoundConstants,
    seed: u64,
) -> f64 {
    let k = k_hat(ka, eta);
    let level = Level::new(k, 1.0);
    let needed: Vec<usize> = {
        let mut c: Vec<usize> = offsets.iter().map(|&o| class_of_offset(o)).collect();
        c.sort_unstable();
        c.dedup();
        c
    };
    let m2l = DiagonalM2l::<T>::build(p, k, &needed);
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
        let reference = evaluate(p, &level, centre, &rcr_local(p, &level, o, &m), &targets);
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
        if measured > e / THETA {
            // At E's floor the comparison measures RCR's rounding too (module documentation):
            // RCR's own error, estimated against the projection, is added.
            let projected = evaluate(
                p,
                &level,
                centre,
                &projected_local(p, &level, o, &m),
                &targets,
            );
            let rcr_error = error(&reference, &projected);
            println!(
                "  {o:?} at the floor: against RCR {measured:.2e} > 2E = {:.2e}; RCR against the \
                 projection {rcr_error:.2e}, the diagonal {:.2e}",
                e / THETA,
                error(&got, &projected)
            );
            if measured > e / THETA + rcr_error {
                failures.push(format!(
                    "{o:?}: {measured:.2e} > 2E + RCR's {rcr_error:.2e} = {:.2e}",
                    e / THETA + rcr_error
                ));
            }
        }
    }
    println!(
        "k a = {ka}, η = {eta}, p = {p}, {} offsets: largest measured/E {worst:.3}",
        offsets.len()
    );
    assert!(failures.is_empty(), "k a = {ka}, p = {p}: {failures:?}");
    worst
}

fn representatives() -> Vec<[i32; 3]> {
    classes().iter().map(|c| c.representative).collect()
}

/// Acceptance (proposal §10 CH4.2): the tables' diagonal M2L against RCR on the 34 classes
/// within E(l, c), for real k at k a = 2.83 (p = 7, S6's worst ratio) and 8, weakly damped k
/// at k a = 8 (Im k = Re k / 10), p from §3.5's formula at 10⁻⁶; k a = 16 in release builds.
#[test]
fn against_rcr_on_the_classes_within_the_bound() {
    let reps = representatives();
    let mut cases = vec![(2.83, 0.0, 7), (8.0, 0.0, formula_degree(8.0, 1e-6))];
    cases.push((8.0, 0.1, formula_degree(8.0, 1e-6)));
    if !cfg!(debug_assertions) {
        cases.push((16.0, 0.0, formula_degree(16.0, 1e-6)));
    }
    for (i, (ka, eta, p)) in cases.into_iter().enumerate() {
        against_rcr::<f64>(ka, eta, p, &reps, BoundConstants::F64, 100 * i as u64);
    }
}

/// The 316 offsets through their classes' maps against RCR within 2E, at k a = 5, p = 16.
#[test]
fn every_offset_against_rcr_within_the_bound() {
    let offsets: Vec<[i32; 3]> = m2l_offsets().iter().map(|o| o.map(|x| x as i32)).collect();
    against_rcr::<f64>(5.0, 0.0, 16, &offsets, BoundConstants::F64, 7000);
}

/// f32 (the tables rounded from f64, the conversions in f32) against RCR in f64 within the
/// f32 bound (C_T = 0.18, C₀ = 50, u = 2⁻²⁴): k a = 8, p = 16; k a ≈ 16, p = 30 in release.
#[test]
fn f32_against_rcr_within_its_bound() {
    let reps = representatives();
    against_rcr::<f32>(8.0, 0.0, 16, &reps, BoundConstants::F32, 9000);
    if !cfg!(debug_assertions) {
        against_rcr::<f32>(16.0, 0.0, 30, &reps, BoundConstants::F32, 9100);
    }
}

/// As the acceptance at large degrees: k a = 32 (the formula at 10⁻⁸), 54 (p = 72, level 3 of
/// W1 at kD = 500 calibrated, and 82, the formula at 10⁻⁶) on the 34 classes, and 108
/// (p = 131, level 2) on the nearest, a middle and the farthest class. Release only.
#[test]
#[ignore = "large degrees: run in release by hand"]
fn against_rcr_at_large_degrees() {
    let reps = representatives();
    for (i, (ka, p)) in [(32.0, formula_degree(32.0, 1e-8)), (54.0, 72), (54.0, 82)]
        .into_iter()
        .enumerate()
    {
        against_rcr::<f64>(ka, 0.0, p, &reps, BoundConstants::F64, 20_000 + i as u64);
    }
    against_rcr::<f64>(
        108.0,
        0.0,
        131,
        &[[0, 0, 2], [1, 2, 2], [3, 3, 3]],
        BoundConstants::F64,
        21_000,
    );
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

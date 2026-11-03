//! The pair chain of the Helmholtz exponential M2L (Helmholtz CH6.1c,
//! `nd_fmm_tables::helmholtz::plane_wave::pair`; CONVENTIONS §3.15, "Exponential M2L")
//! against RCR (`HelmholtzTables::m2l`), with FMM3D's quadratures (CH6.1a) and with
//! `nd_fmm_ref`'s generated ones (CH6.1b).
//!
//! Error measure: per level the multipole of point charges in the source box (its corners,
//! its centre and seeded points, seeded charges with Σ|q| = 1, P2M in stored form) is
//! translated to every V offset both ways, and both locals are evaluated by L2P at the 27
//! points {−½, 0, ½}³ (box widths) of the target box; the error is the largest
//! |Δ| over them in box widths, twice the level units' r_l φ̃ (the quadrature's absolute
//! norm, S4 §1, in which FMM3D states its errors). The bound is [`CHAIN_BOUND`] times the
//! stated error of FMM3D's quadrature, or times the ε of a generated one.

use nd_fmm_math::orthonormal::{harmonics, legendre};
use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::plane_wave::generate;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::plane_wave::pair::{
    Nodes, PlaneWaveLevel, harmonic_factors, shift_add,
};
use nd_fmm_tables::helmholtz::plane_wave::{Precision, Quadrature};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_tables::plane_wave::geometry::{Direction, slot_of};

type C = Complex<f64>;

/// The bound of the chain against RCR, in units of the quadrature's error (the module
/// documentation): CH6.1a's bound of the point-to-point chain.
const CHAIN_BOUND: f64 = 3.0;

/// A seeded generator of uniform values in [−½, ½).
struct Seeded(u64);

impl Seeded {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// The sources (level units, half-width 1, the source box at the origin): the corners, the
/// centre and four seeded points; seeded charges, real if `real`, with Σ|q| = 1.
fn sources(real: bool, seed: u64) -> (Vec<[f64; 3]>, Vec<C>) {
    let mut rng = Seeded::new(seed);
    let mut points: Vec<[f64; 3]> = (0..8)
        .map(|c| [c & 4, c & 2, c & 1].map(|b| if b == 0 { -1.0 } else { 1.0 }))
        .collect();
    points.push([0.0; 3]);
    for _ in 0..4 {
        points.push([(); 3].map(|_| 2.0 * rng.next()));
    }
    let mut charges: Vec<C> = points
        .iter()
        .map(|_| {
            let im = if real { 0.0 } else { rng.next() };
            C::new(rng.next() + 0.75, im)
        })
        .collect();
    let total: f64 = charges.iter().map(|q| q.norm()).sum();
    for q in &mut charges {
        *q /= total;
    }
    (points, charges)
}

/// The 27 targets of the box at the offset `d` (level units).
fn targets(d: [i64; 3]) -> Vec<[f64; 3]> {
    (0..27)
        .map(|i| {
            let e = [i / 9, (i / 3) % 3, i % 3];
            core::array::from_fn(|a| 2.0 * d[a] as f64 + (e[a] as f64 - 1.0))
        })
        .collect()
}

/// The directions of the pair at the offset index `index`: [`Direction::of`] its offset
/// and those that merge and shift gives it at any child ([`Direction::for_slot`]).
fn directions(index: usize) -> Vec<Direction> {
    let d = m2l_offsets()[index];
    let mut out = vec![Direction::of(d).expect("a V offset has a direction")];
    for child in 0..8 {
        if let Some(slot) = slot_of(child, d)
            && let Some(direction) = Direction::for_slot(slot, child)
            && !out.contains(&direction)
        {
            out.push(direction);
        }
    }
    out
}

/// One case: a level of `rcr` with its exponential M2L.
struct Case<'a> {
    rcr: &'a HelmholtzTables<f64>,
    level: usize,
    waves: &'a PlaneWaveLevel<f64>,
}

/// The largest error (the module documentation) over the offsets `offsets` and their
/// [`directions`], with `planes` multipole planes (1: real charges, real k).
fn pair_error(case: &Case<'_>, planes: usize, offsets: impl Iterator<Item = usize>) -> f64 {
    let Case { rcr, level, waves } = *case;
    let p = rcr.degree(level);
    let len = Layout::new(p).len();
    let k_level = Level::new(rcr.k_hat(level), 1.0);
    let mut ws = Workspace::new(p);
    let (points, charges) = sources(planes == 1, level as u64);
    let mut multipole = vec![0.0; 2 * len];
    leaf::p2m(
        p,
        &k_level,
        &Frame::new([0.0; 3], 1.0),
        &points,
        &charges,
        &mut ws,
        &mut multipole,
    );
    if planes == 1 {
        assert!(multipole[len..].iter().all(|&v| v == 0.0));
        multipole.truncate(len);
    }
    let mut rcr_scratch = rcr.scratch();
    let mut scratch = waves.scratch();
    let mut worst = 0.0f64;
    for index in offsets {
        let d = m2l_offsets()[index];
        let to = Frame::new(d.map(|x| 2.0 * x as f64), 1.0);
        let targets = targets(d);
        let mut expected = vec![0.0; 2 * len];
        rcr.m2l(
            level,
            index,
            &multipole,
            planes,
            &mut expected,
            1,
            &mut rcr_scratch,
        );
        let mut reference = vec![C::new(0.0, 0.0); targets.len()];
        leaf::l2p(
            p,
            &k_level,
            &to,
            &expected,
            &targets,
            &mut ws,
            &mut reference,
            None,
        );
        for direction in directions(index) {
            let mut local = vec![0.0; 2 * len];
            waves.m2l(
                index,
                direction,
                &multipole,
                planes,
                &mut local,
                &mut scratch,
            );
            let mut got = vec![C::new(0.0, 0.0); targets.len()];
            leaf::l2p(p, &k_level, &to, &local, &targets, &mut ws, &mut got, None);
            for (a, b) in got.iter().zip(&reference) {
                worst = worst.max(2.0 * (*a - *b).norm());
            }
        }
    }
    worst
}

/// The plane counts of a level: one (real charges) as well as two for real k.
fn plane_counts(rcr: &HelmholtzTables<f64>) -> &'static [usize] {
    if rcr.real_shifts() { &[1, 2] } else { &[2] }
}

/// The degrees of levels 0..`levels` for k̂₀ = k r₀: [`resolved`] from `base` on levels
/// ≥ 2, where κ_l = 2k̂₀/2^l, and 0 above (no M2L).
fn degrees(k_root: C, levels: usize, base: usize) -> Vec<usize> {
    (0..levels)
        .map(|l| {
            if l < 2 {
                0
            } else {
                resolved(base, k_root * 2.0 * 0.5f64.powi(l as i32))
            }
        })
        .collect()
}

/// Checks levels `levels` of k̂₀ at the degrees [`resolved`] from `base` with FMM3D's
/// quadratures at `precision`, on the offsets `step` apart: each level's error within
/// [`CHAIN_BOUND`] times its domain's stated error.
fn check_fmm3d(
    k_root: C,
    levels: core::ops::Range<usize>,
    base: usize,
    precision: Precision,
    step: usize,
) {
    let degrees = degrees(k_root, levels.end, base);
    let rcr = HelmholtzTables::<f64>::build(k_root, &degrees, DEFAULT_BLOCK_BUDGET);
    for level in levels {
        let p = rcr.degree(level);
        let k_hat = rcr.k_hat(level);
        let quadrature = Quadrature::new(2.0 * k_hat, precision).expect("κ in the tables");
        let waves = PlaneWaveLevel::<f64>::new(k_hat, p, &quadrature);
        let case = Case {
            rcr: &rcr,
            level,
            waves: &waves,
        };
        for &planes in plane_counts(&rcr) {
            let error = pair_error(&case, planes, (0..M2L_OFFSET_COUNT).step_by(step));
            let ratio = error / quadrature.stated_error();
            println!(
                "k̂₀ = {k_root}, level {level} (κ = {:.4}), {precision:?}, p = {p}, {planes} \
                 plane(s): {error:.2e}, {ratio:.2} × the stated {:.1e}",
                2.0 * k_hat,
                quadrature.stated_error()
            );
            assert!(ratio <= CHAIN_BOUND, "level {level}: {ratio:.2}×");
        }
    }
}

/// As [`check_fmm3d`] on one level with the quadrature generated for ε, the bound
/// [`CHAIN_BOUND`] ε.
fn check_generated(k_root: C, level: usize, base: usize, eps: f64, step: usize) {
    let rcr = HelmholtzTables::<f64>::build(
        k_root,
        &degrees(k_root, level + 1, base),
        DEFAULT_BLOCK_BUDGET,
    );
    let p = rcr.degree(level);
    let k_hat = rcr.k_hat(level);
    let q = generate(2.0 * k_hat, eps);
    assert!(q.shortfall().is_none(), "the generator met ε");
    let nodes = Nodes {
        lambda: q.lambda(),
        t: q.t(),
        weights: q.weights(),
        alpha: q.alpha_counts(),
        fourier: q.fourier_counts(),
    };
    let waves = PlaneWaveLevel::<f64>::from_nodes(k_hat, p, nodes);
    let case = Case {
        rcr: &rcr,
        level,
        waves: &waves,
    };
    for &planes in plane_counts(&rcr) {
        let error = pair_error(&case, planes, (0..M2L_OFFSET_COUNT).step_by(step));
        println!(
            "k̂₀ = {k_root}, level {level} (κ = {:.4}), generated at ε = {eps:.0e}, p = {p}, \
             {planes} plane(s): {error:.2e}, {:.2} ε",
            2.0 * k_hat,
            error / eps
        );
        assert!(
            error <= CHAIN_BOUND * eps,
            "level {level}: {:.2} ε",
            error / eps
        );
    }
}

/// The wavenumbers k̂₀ = k r₀: real, complex and strongly damped. Levels 2 to 9 give
/// κ_l = 2k̂₀/2^l from 40 (k̂₀ = 80, level 2) to about 0.3 (level 9).
const K_ROOTS: [(f64, f64); 3] = [(80.0, 0.0), (80.0, 12.0), (20.0, 40.0)];

/// The precisions of FMM3D held here with the base of [`resolved`].
const PRECISIONS: [(Precision, usize); 3] = [
    (Precision::Iprec1, 20),
    (Precision::Iprec2, 36),
    (Precision::Iprec3, 60),
];

/// A degree that resolves the multipoles of the tests' charges on a level of κ from a
/// precision's `base` (its degree at low κ), base + ⌈0.9 |κ|⌉, so that the chain's
/// difference from RCR is the quadrature's error alone: below it the difference also holds
/// the quadrature's error on the truncated multipole's tail ([`degree_study`]).
fn resolved(base: usize, kappa: C) -> usize {
    base + (0.9 * kappa.norm()).ceil() as usize
}

#[test]
fn harmonic_factors_are_the_harmonics_at_real_points() {
    let p = 12;
    let layout = Layout::new(p);
    let mut rng = Seeded::new(5);
    for _ in 0..20 {
        let x = [rng.next(), rng.next(), rng.next()];
        let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
        let rho = x[0].hypot(x[1]);
        let phi = x[1].atan2(x[0]);
        let mut y = vec![0.0; layout.len()];
        harmonics(p, x, &mut y);
        let mut f = vec![C::new(0.0, 0.0); legendre::len(p)];
        harmonic_factors(
            p,
            C::new(x[2], 0.0),
            C::new(rho, 0.0),
            C::new(r * r, 0.0),
            &mut f,
        );
        for n in 0..=p {
            for m in 0..=n {
                let v = f[legendre::index(n, m)] / r.powi(n as i32);
                assert!(v.im == 0.0);
                let (s, c) = (m as f64 * phi).sin_cos();
                let plus = y[layout.idx(n, m as isize)];
                assert!(
                    (v.re * c - plus).abs() <= 1e-13,
                    "({n}, {m}): {} vs {plus}",
                    v.re * c
                );
                if m > 0 {
                    let minus = y[layout.idx(n, -(m as isize))];
                    assert!((v.re * s - minus).abs() <= 1e-13, "({n}, −{m})");
                }
            }
        }
    }
}

/// Homogeneity at a complex point: the factors of degree n scale by cⁿ when z, ρ and r² do
/// by c, c and c² (they are polynomials, finite where r² = 0).
#[test]
fn harmonic_factors_are_homogeneous() {
    let p = 10;
    let (z, rho) = (C::new(0.3, 1.7), C::new(1.2, -0.4));
    let r2 = z * z + rho * rho;
    let c = C::new(0.6, 0.8) * 1.5;
    let (mut a, mut b) = (
        vec![C::new(0.0, 0.0); legendre::len(p)],
        vec![C::new(0.0, 0.0); legendre::len(p)],
    );
    harmonic_factors(p, z, rho, r2, &mut a);
    harmonic_factors(p, c * z, c * rho, c * c * r2, &mut b);
    for n in 0..=p {
        for m in 0..=n {
            let i = legendre::index(n, m);
            let expected = a[i] * c.powi(n as i32);
            assert!(
                (b[i] - expected).norm() <= 1e-12 * expected.norm().max(1.0),
                "({n}, {m})"
            );
        }
    }
}

/// FMM3D's quadratures, debug: each wavenumber at two precisions, each on one level (a high
/// and a low κ), every seventh offset.
#[test]
fn against_rcr_with_fmm3ds_quadratures() {
    for (k_root, cases) in [
        (
            (80.0, 0.0),
            [(Precision::Iprec1, 3), (Precision::Iprec2, 7)],
        ),
        (
            (80.0, 12.0),
            [(Precision::Iprec3, 7), (Precision::Iprec1, 5)],
        ),
        (
            (20.0, 40.0),
            [(Precision::Iprec2, 5), (Precision::Iprec1, 3)],
        ),
    ] {
        for (precision, level) in cases {
            let base = PRECISIONS[precision.iprec() - 1].1;
            let k_root = C::new(k_root.0, k_root.1);
            check_fmm3d(k_root, level..level + 1, base, precision, 7);
        }
    }
}

/// FMM3D's quadratures on every offset of levels 2 to 9 (release).
#[test]
#[ignore = "release: every offset on eight levels at three precisions"]
fn against_rcr_with_fmm3ds_quadratures_every_level() {
    for (re, im) in K_ROOTS {
        for (precision, base) in PRECISIONS {
            check_fmm3d(C::new(re, im), 2..10, base, precision, 1);
        }
    }
}

/// Generated quadratures, debug: two levels.
#[test]
fn against_rcr_with_generated_quadratures() {
    check_generated(C::new(80.0, 12.0), 5, 20, 1e-3, 5);
    check_generated(C::new(80.0, 0.0), 8, 36, 1e-6, 5);
}

/// Generated quadratures on every offset (release).
#[test]
#[ignore = "release: the generator at nine levels and ε, every offset"]
fn against_rcr_with_generated_quadratures_every_level() {
    for (re, im) in K_ROOTS {
        for (eps, base) in [(1e-3, 20), (1e-6, 36), (1e-9, 60)] {
            for level in [2, 5, 9] {
                check_generated(C::new(re, im), level, base, eps, 1);
            }
        }
    }
}

/// A shift by d₁ then d₂ is one by d₁ + d₂, to rounding.
#[test]
fn shifts_compose() {
    let k_hat = C::new(3.0, 0.4);
    let q = Quadrature::new(2.0 * k_hat, Precision::Iprec2).expect("κ in the tables");
    let waves = PlaneWaveLevel::<f64>::new(k_hat, 10, &q);
    let w = waves.wave_len();
    let mut rng = Seeded::new(3);
    let x: Vec<f64> = (0..w).map(|_| rng.next()).collect();
    for up in [true, false] {
        let (d1, d2) = ([0.5, -1.5, 2.5], [-0.5, 0.5, -0.5]);
        let sum = [0.0, -1.0, 2.0];
        let (mut f1, mut f2, mut f) = (vec![0.0; w], vec![0.0; w], vec![0.0; w]);
        waves.shift_factors(d1, up, &mut f1);
        waves.shift_factors(d2, up, &mut f2);
        waves.shift_factors(sum, up, &mut f);
        let (mut a, mut b) = (vec![0.0; w], vec![0.0; w]);
        let mut once = vec![0.0; w];
        shift_add(&mut a, &x, &f1);
        shift_add(&mut b, &a, &f2);
        shift_add(&mut once, &x, &f);
        let largest = once.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (u, v) in b.iter().zip(&once) {
            assert!((u - v).abs() <= 1e-13 * largest, "up = {up}");
        }
    }
}

/// The frames' rotations are orthogonal: into a frame and back is the identity.
#[test]
fn frames_round_trip() {
    let k_hat = C::new(1.0, 0.0);
    let q = Quadrature::new(2.0 * k_hat, Precision::Iprec1).expect("κ in the tables");
    let p = 9;
    let waves = PlaneWaveLevel::<f64>::new(k_hat, p, &q);
    let len = Layout::new(p).len();
    let mut rng = Seeded::new(9);
    let x: Vec<f64> = (0..2 * len).map(|_| rng.next()).collect();
    for frame in 0..3 {
        let mut rotated = vec![0.0; 2 * len];
        waves.rotate_multipole(frame, &x, 2, &mut rotated);
        let mut back = vec![0.0; 2 * len];
        waves.add_local_from_frame(frame, &rotated, &mut back);
        for (a, b) in back.iter().zip(&x) {
            assert!((a - b).abs() <= 1e-14, "frame {frame}");
        }
    }
}

/// The waves represent r_l φ̃ of the multipole (the convention): at a point above the source
/// box, Σ X e^{iκŝ·u} against M2P, within the quadrature's error.
#[test]
fn up_waves_represent_the_multipole() {
    let k_hat = C::new(2.5, 0.3);
    let p = 14;
    let q = Quadrature::new(2.0 * k_hat, Precision::Iprec2).expect("κ in the tables");
    let waves = PlaneWaveLevel::<f64>::new(k_hat, p, &q);
    let level = Level::new(k_hat, 1.0);
    let len = Layout::new(p).len();
    let mut ws = Workspace::new(p);
    let (points, charges) = sources(false, 1);
    let mut multipole = vec![0.0; 2 * len];
    leaf::p2m(
        p,
        &level,
        &Frame::new([0.0; 3], 1.0),
        &points,
        &charges,
        &mut ws,
        &mut multipole,
    );
    let w = waves.wave_len();
    let (mut up, mut down) = (vec![0.0; w], vec![0.0; w]);
    let mut scratch = waves.scratch();
    waves.waves_of(0, &multipole, 2, &mut up, &mut down, &mut scratch);
    // Points (box widths) at 2 to 3 boxes above and below.
    for u in [[0.3, -0.2, 2.0], [1.5, 2.5, 3.0], [-3.5, 0.5, 2.5]] {
        for (sign, x) in [(1.0, &up), (-1.0, &down)] {
            let point = [2.0 * u[0], 2.0 * u[1], 2.0 * sign * u[2]];
            let mut expected = [C::new(0.0, 0.0)];
            leaf::m2p(
                p,
                &level,
                &Frame::new([0.0; 3], 1.0),
                &multipole,
                &[point],
                &mut ws,
                &mut expected,
                None,
            );
            let got = waves.evaluate(x, [u[0], u[1], sign * u[2]], sign > 0.0);
            let error = 2.0 * (got - expected[0]).norm();
            assert!(
                error <= CHAIN_BOUND * q.stated_error(),
                "u = {u:?}, sign {sign}: {error:e} of {:e}",
                q.stated_error()
            );
        }
    }
}

#[test]
#[should_panic(expected = "not 2 or 3 boxes")]
fn a_pair_off_the_region_is_refused() {
    let k_hat = C::new(1.0, 0.0);
    let q = Quadrature::new(2.0 * k_hat, Precision::Iprec1).expect("κ in the tables");
    let waves = PlaneWaveLevel::<f64>::new(k_hat, 4, &q);
    let len = Layout::new(4).len();
    let mut scratch = waves.scratch();
    // Offset index 0 is (−3, −3, −3): not along +z.
    let mut local = vec![0.0; 2 * len];
    waves.m2l(
        0,
        Direction::Up,
        &vec![0.0; 2 * len],
        2,
        &mut local,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "κ")]
fn a_quadrature_of_another_level_is_refused() {
    let q = Quadrature::new(C::new(2.0, 0.0), Precision::Iprec1).expect("κ in the tables");
    let _ = PlaneWaveLevel::<f64>::new(C::new(2.0, 0.0), 4, &q);
}

/// The study behind [`PRECISIONS`]' degrees (no assertion; docs/helmholtz/tasks/CH6.1c.md,
/// "Results"): on every third offset along [`Direction::of`], with two planes, the largest
/// error of the chain and of RCR against the direct sum Σ q e^{ik̂r}/r at the targets, in
/// box widths, at every precision and degree 8 to 40.
#[test]
#[ignore = "release: a study, prints its table"]
fn degree_study() {
    println!("| κ | iprec | stated | p | chain − RCR | chain − direct | RCR − direct |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for (re, im, level) in [
        (80.0, 0.0, 4),
        (80.0, 0.0, 7),
        (80.0, 0.0, 9),
        (80.0, 12.0, 7),
        (20.0, 40.0, 6),
    ] {
        for p in [8, 12, 16, 20, 24, 28, 32, 40] {
            for precision in Precision::ALL {
                study(C::new(re, im), level, p, precision);
            }
        }
    }
}

/// One row of [`degree_study`].
fn study(k_root: C, level: usize, p: usize, precision: Precision) {
    let degrees: Vec<usize> = (0..=level).map(|l| if l < 2 { 0 } else { p }).collect();
    let rcr = HelmholtzTables::<f64>::build(k_root, &degrees, DEFAULT_BLOCK_BUDGET);
    let k_hat = rcr.k_hat(level);
    let quadrature = Quadrature::new(2.0 * k_hat, precision).expect("κ in the tables");
    let waves = PlaneWaveLevel::<f64>::new(k_hat, p, &quadrature);
    let len = Layout::new(p).len();
    let k_level = Level::new(k_hat, 1.0);
    let mut ws = Workspace::new(p);
    let (points, charges) = sources(false, level as u64);
    let mut multipole = vec![0.0; 2 * len];
    leaf::p2m(
        p,
        &k_level,
        &Frame::new([0.0; 3], 1.0),
        &points,
        &charges,
        &mut ws,
        &mut multipole,
    );
    let (mut rcr_scratch, mut scratch) = (rcr.scratch(), waves.scratch());
    let mut worst = [0.0f64; 3];
    for index in (0..M2L_OFFSET_COUNT).step_by(3) {
        let d = m2l_offsets()[index];
        let to = Frame::new(d.map(|x| 2.0 * x as f64), 1.0);
        let targets = targets(d);
        let (mut a, mut b) = (vec![0.0; 2 * len], vec![0.0; 2 * len]);
        rcr.m2l(level, index, &multipole, 2, &mut a, 1, &mut rcr_scratch);
        waves.m2l_of_offset(index, &multipole, 2, &mut b, &mut scratch);
        let mut values = [vec![C::new(0.0, 0.0); 27], vec![C::new(0.0, 0.0); 27]];
        for (local, out) in [&a, &b].into_iter().zip(&mut values) {
            leaf::l2p(p, &k_level, &to, local, &targets, &mut ws, out, None);
        }
        for (t, x) in targets.iter().enumerate() {
            let direct: C = points
                .iter()
                .zip(&charges)
                .map(|(y, q)| {
                    let r = (0..3).map(|a| (x[a] - y[a]).powi(2)).sum::<f64>().sqrt();
                    *q * (C::i() * k_hat * r).exp() / r
                })
                .sum();
            let (rcr_value, chain) = (values[0][t], values[1][t]);
            for (w, e) in
                worst
                    .iter_mut()
                    .zip([chain - rcr_value, chain - direct, rcr_value - direct])
            {
                *w = w.max(2.0 * e.norm());
            }
        }
    }
    println!(
        "| {:.4} | {} | {:.1e} | {p} | {:.1e} | {:.1e} | {:.1e} |",
        2.0 * k_hat,
        precision.iprec(),
        quadrature.stated_error(),
        worst[0],
        worst[1],
        worst[2]
    );
}

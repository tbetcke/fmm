//! The diagonal form of the M2L (Helmholtz CH1.3d1, `nd_fmm_ref::helmholtz::diagonal`;
//! proposal §3.6 and §5.4, `docs/helmholtz/tasks/CH1.3d1.md`).
//!
//! The grid's quadrature, S → F → R, the signature of a charge, the permuted transfer
//! functions, and the acceptance: `diagonal::m2l` against `rcr::m2l` (CH1.3c) within the
//! bound E of proposal §3.6. As S6 measured (`spikes/diagonal/SPIKE_REPORT.md` §1.2), the
//! error of an M2L is the larger of the relative ℓ2 errors over the targets of φ̃ and of
//! ∇φ̃ after `leaf::l2p` (proposal §9.1), with complex charges uniform in the source box
//! and at its corners, and targets uniform in the target box and at its corners. "Within
//! E" is measured ≤ 2E = E/θ: S6 fitted C_T so that the measurement stays within 2E, and
//! the rule E ≤ θ ε then bounds the error by ε (proposal §3.6).

use nd_fmm_math::orthonormal::harmonics;
use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::diagonal::{
    self, BoundConstants, Grid, THETA, Transfer, bound, class_of, classes, offsets,
};
use nd_fmm_ref::helmholtz::{Level, Workspace, direct, leaf, p2p, rcr};

use crate::common::{C, SplitMix64, Worst, rel_l2, rel_l2_gradient};
use crate::leaf_common::Field;
use crate::translation_common::in_box;

/// The boxes of the tests: width w = 1 (half-width 1/2), the source at the origin.
const HALF: f64 = 0.5;

fn circumradius() -> f64 {
    3.0_f64.sqrt() * HALF
}

/// k with k a = `ka` on the test level; `eta` = Im k / Re k.
fn wavenumber(ka: f64, eta: f64) -> C {
    C::new(ka / circumradius(), eta * ka / circumradius())
}

fn source() -> Frame<f64> {
    Frame::new([0.0; 3], HALF)
}

fn target(offset: [i32; 3]) -> Frame<f64> {
    Frame::new(offset.map(|o| o as f64), HALF)
}

/// The degree of proposal §3.5's formula (Gumerov and Duraiswami 2009, δ = 2):
/// p = ⌈P⌉ − 1 with P = (p_lo⁴ + p_hi⁴)^{1/4}.
fn formula_degree(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.cbrt();
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2f64.ln();
    let big_p = (p_lo.powi(4) + p_hi.powi(4)).powf(0.25);
    big_p.ceil() as usize - 1
}

/// Seeded sources, charges and targets for an offset.
struct Case {
    sources: Vec<[f64; 3]>,
    charges: Vec<C>,
    targets: Vec<[f64; 3]>,
}

impl Case {
    fn new(seed: u64, offset: [i32; 3]) -> Self {
        let mut rng = SplitMix64::new(seed);
        let sources = in_box(&mut rng, 120, &source(), false);
        let charges = rng.complex_charges(sources.len());
        let targets = in_box(&mut rng, 56, &target(offset), false);
        Self {
            sources,
            charges,
            targets,
        }
    }
}

fn multipole(p: usize, level: &Level<f64>, case: &Case) -> Vec<f64> {
    let mut ws = Workspace::new(p);
    let mut out = vec![0.0; 2 * Layout::new(p).len()];
    leaf::p2m(
        p,
        level,
        &source(),
        &case.sources,
        &case.charges,
        &mut ws,
        &mut out,
    );
    out
}

fn evaluate(
    p: usize,
    level: &Level<f64>,
    offset: [i32; 3],
    local: &[f64],
    x: &[[f64; 3]],
) -> Field {
    let mut ws = Workspace::new(p);
    let mut field = Field::zero(x.len());
    leaf::l2p(
        p,
        level,
        &target(offset),
        local,
        x,
        &mut ws,
        &mut field.potential,
        Some(&mut field.gradient),
    );
    field
}

/// max(relative ℓ2 of φ̃, relative ℓ2 of ∇φ̃) of `a` against `b`.
fn error(a: &Field, b: &Field) -> f64 {
    rel_l2(&a.potential, &b.potential).max(rel_l2_gradient(&a.gradient, &b.gradient))
}

fn rcr_local(p: usize, level: &Level<f64>, offset: [i32; 3], m: &[f64]) -> Vec<f64> {
    let (from, to) = (source(), target(offset));
    let table = rcr::m2l_table(p, p, level, &from, &to);
    let mut ws = Workspace::new(p);
    let mut local = vec![0.0; m.len()];
    rcr::m2l(p, p, level, &from, &to, &table, &mut ws, m, &mut local);
    local
}

fn diagonal_local(
    grid: &Grid<f64>,
    level: &Level<f64>,
    offset: [i32; 3],
    transfer: &Transfer<f64>,
    m: &[f64],
) -> Vec<f64> {
    let mut ws = diagonal::Workspace::new(grid.degree());
    let mut local = vec![0.0; m.len()];
    diagonal::m2l(
        grid,
        level,
        &source(),
        &target(offset),
        transfer,
        &mut ws,
        m,
        &mut local,
    );
    local
}

fn distance(o: [i32; 3]) -> f64 {
    (o.iter().map(|&x| (x * x) as f64).sum::<f64>()).sqrt()
}

/// The quadrature is exact to degree 4p: Yₙᵐ are orthonormal on the grid for n ≤ 2p.
#[test]
fn grid_quadrature_is_exact() {
    for p in [1, 2, 5, 10] {
        let grid = Grid::<f64>::new(p);
        assert_eq!(grid.azimuths() % 4, 0);
        assert_eq!(grid.rings(), 2 * p + 1);
        for i in 0..grid.rings() {
            assert_eq!(grid.node(grid.rings() - 1 - i), -grid.node(i));
            if i > 0 {
                assert!(grid.node(i) > grid.node(i - 1));
            }
        }
        let layout = Layout::new(2 * p);
        let mut y = vec![0.0; layout.len()];
        let mut gram = vec![0.0; layout.len() * layout.len()];
        for i in 0..grid.rings() {
            for j in 0..grid.azimuths() {
                harmonics(2 * p, grid.direction(i, j), &mut y);
                let w = grid.weight(i);
                for a in 0..y.len() {
                    for b in 0..y.len() {
                        gram[a * y.len() + b] += w * y[a] * y[b];
                    }
                }
            }
        }
        let mut worst: f64 = 0.0;
        for a in 0..y.len() {
            for b in 0..y.len() {
                let expected = if a == b { 1.0 } else { 0.0 };
                worst = worst.max((gram[a * y.len() + b] - expected).abs());
            }
        }
        assert!(worst < 1e-13, "p = {p}: Gram error {worst:e}");
    }
}

/// f2r(s2f(M̂)) = s²ⁿ M̂: the transforms invert each other on degrees n ≤ p.
#[test]
fn s2f_then_f2r_is_the_identity() {
    for (ka, p) in [(5.0, 9), (0.3, 9), (20.0, 24)] {
        let level = Level::new(wavenumber(ka, 0.0), HALF);
        let s = level.scale();
        let grid = Grid::new(p);
        let mut ws = diagonal::Workspace::new(p);
        let mut rng = SplitMix64::new(17 + p as u64);
        let len = Layout::new(p).len();
        let m: Vec<f64> = (0..2 * len).map(|_| rng.range(-1.0, 1.0)).collect();
        let mut signature = vec![C::new(0.0, 0.0); grid.samples()];
        diagonal::s2f(&grid, &level, &m, &mut ws, &mut signature);
        let mut back = vec![0.0; 2 * len];
        diagonal::f2r(&grid, &level, &signature, &mut ws, &mut back);
        let mut worst: f64 = 0.0;
        for n in 0..=p {
            let f = s.powi(2 * n as i32);
            for i in n * n..(n + 1) * (n + 1) {
                for plane in 0..2 {
                    let at = plane * len + i;
                    worst = worst.max((back[at] - f * m[at]).abs());
                }
            }
        }
        assert!(worst < 1e-13, "k a = {ka}, p = {p}: {worst:e}");
    }
}

/// For one charge q at y in the box, 4π e^β F̂(ŝ) = q e^{−ik ŝ·(y − c)} to the truncation
/// (s = 1): the signature's normalisation.
#[test]
fn the_signature_of_a_charge() {
    let p = 30;
    for eta in [0.0, 0.1] {
        let k = wavenumber(8.0, eta);
        let level = Level::new(k, HALF);
        let y = [0.31, -0.22, 0.4];
        let q = C::new(0.7, -0.3);
        let mut ws = Workspace::new(p);
        let mut m = vec![0.0; 2 * Layout::new(p).len()];
        leaf::p2m(p, &level, &source(), &[y], &[q], &mut ws, &mut m);
        let grid = Grid::new(p);
        let mut signature = vec![C::new(0.0, 0.0); grid.samples()];
        diagonal::s2f(
            &grid,
            &level,
            &m,
            &mut diagonal::Workspace::new(p),
            &mut signature,
        );
        let factor = 4.0 * std::f64::consts::PI * level.damping().exp();
        let mut worst: f64 = 0.0;
        for i in 0..grid.rings() {
            for j in 0..grid.azimuths() {
                let s = grid.direction(i, j);
                let dot = s[0] * y[0] + s[1] * y[1] + s[2] * y[2];
                let exact = q * (C::new(0.0, -1.0) * k * dot).exp();
                let got = signature[i * grid.azimuths() + j] * factor;
                worst = worst.max((got - exact).norm() / exact.norm());
            }
        }
        assert!(worst < 1e-12, "η = {eta}: {worst:e}");
    }
}

/// Each offset's transfer function read through its class representative's equals the
/// one computed for the offset itself, to an ulp of the largest sample.
#[test]
fn permuted_transfer_functions_are_direct_ones() {
    let p = 10;
    let level = Level::new(wavenumber(5.0, 0.1), HALF);
    let grid = Grid::new(p);
    let reps: Vec<([i32; 3], Transfer<f64>)> = classes()
        .iter()
        .map(|c| {
            let rep = c.representative;
            (rep, Transfer::new(&grid, &level, &source(), &target(rep)))
        })
        .collect();
    for o in offsets() {
        let (rep, symmetry) = class_of(o);
        let base = &reps.iter().find(|(r, _)| *r == rep).unwrap().1;
        let permuted = base.permuted(&grid, symmetry);
        let direct = Transfer::new(&grid, &level, &source(), &target(o));
        assert_eq!(permuted.shift(), direct.shift(), "offset {o:?}");
        let scale = direct.samples().iter().fold(0.0f64, |a, z| a.max(z.norm()));
        let worst = permuted
            .samples()
            .iter()
            .zip(direct.samples())
            .fold(0.0f64, |a, (x, y)| a.max((x - y).norm()));
        assert!(
            worst <= 2.0 * f64::EPSILON * scale,
            "offset {o:?}: {worst:e}"
        );
    }
}

/// One (k a, η, p) of the acceptance: the 34 class representatives against RCR within
/// 2E; returns the largest ratio measured/E and the number of classes with E < 10⁻⁶.
/// With `DIAGONAL_DETAIL` set it prints every class.
fn against_rcr_on_the_classes(ka: f64, eta: f64, p: usize, seed: u64) -> (f64, usize) {
    let k = wavenumber(ka, eta);
    let level = Level::new(k, HALF);
    let grid = Grid::new(p);
    let mut worst_ratio: f64 = 0.0;
    let mut tight = 0;
    let mut failures = Vec::new();
    for class in classes() {
        let o = class.representative;
        let case = Case::new(seed ^ (o[0] * 49 + o[1] * 7 + o[2]) as u64, o);
        let m = multipole(p, &level, &case);
        let reference = evaluate(p, &level, o, &rcr_local(p, &level, o, &m), &case.targets);
        let transfer = Transfer::new(&grid, &level, &source(), &target(o));
        let got = evaluate(
            p,
            &level,
            o,
            &diagonal_local(&grid, &level, o, &transfer, &m),
            &case.targets,
        );
        let measured = error(&got, &reference);
        let e = bound(k, distance(o), circumradius(), p, BoundConstants::F64);
        assert!(measured.is_finite(), "class {o:?}: not finite");
        worst_ratio = worst_ratio.max(measured / e);
        if std::env::var_os("DIAGONAL_DETAIL").is_some() {
            println!(
                "  {o:?}: measured {measured:.3e}, E {e:.3e}, ratio {:.3}",
                measured / e
            );
        }
        if e < 1e-6 {
            tight += 1;
        }
        if measured > e / THETA {
            failures.push(format!("{o:?}: {measured:.2e} > 2E = {:.2e}", e / THETA));
        }
    }
    println!(
        "k a = {ka}, η = {eta}, p = {p}: largest measured/E {worst_ratio:.3}, {tight} of 34 classes with E < 1e-6"
    );
    assert!(failures.is_empty(), "k a = {ka}, p = {p}: {failures:?}");
    (worst_ratio, tight)
}

/// Acceptance (roadmap CH1.3d1): the diagonal M2L against CH1.3c's RCR on all 34 classes
/// within the bound E of proposal §3.6, for real k at k a ≈ 2.83 (p = 7, S6's worst
/// ratio), 8 and 16, and weakly damped k at k a = 8, p from §3.5's formula at 10⁻⁶ and
/// 10⁻⁸ where the debug build affords it.
#[test]
fn m2l_against_rcr_within_the_bound() {
    let mut cases = vec![(2.83, 0.0, 7), (8.0, 0.0, formula_degree(8.0, 1e-6))];
    cases.push((8.0, 0.1, formula_degree(8.0, 1e-6)));
    cases.push((16.0, 0.0, formula_degree(16.0, 1e-6)));
    if !cfg!(debug_assertions) {
        cases.push((8.0, 0.0, formula_degree(8.0, 1e-8)));
        cases.push((16.0, 0.0, formula_degree(16.0, 1e-8)));
    }
    let mut tight = 0;
    for (i, (ka, eta, p)) in cases.into_iter().enumerate() {
        tight += against_rcr_on_the_classes(ka, eta, p, 1000 + i as u64).1;
    }
    // The cases include classes where the bound is tight enough to test.
    assert!(tight >= 34, "only {tight} classes with E < 1e-6");
}

/// As [`m2l_against_rcr_within_the_bound`] on levels 3 and 2 of W1 at kD = 500 (proposal
/// §3.6: k a = 54 and 108... here 54, with p = 72, the calibrated degree, and 82, the
/// formula's at 10⁻⁶), and k a = 32 at the formula's degree at 10⁻⁸. Release only.
#[test]
#[ignore = "large degrees: run in release by hand"]
fn m2l_against_rcr_within_the_bound_at_large_degrees() {
    for (i, (ka, p)) in [(32.0, formula_degree(32.0, 1e-8)), (54.0, 72), (54.0, 82)]
        .into_iter()
        .enumerate()
    {
        against_rcr_on_the_classes(ka, 0.0, p, 2000 + i as u64);
    }
}

/// All 316 offsets through their class's permuted transfer function, against RCR within
/// 2E, at k a = 5 and p = 16.
#[test]
fn every_offset_against_rcr() {
    let (ka, p) = (5.0, 16);
    let k = wavenumber(ka, 0.0);
    let level = Level::new(k, HALF);
    let grid = Grid::new(p);
    let reps: Vec<([i32; 3], Transfer<f64>)> = classes()
        .iter()
        .map(|c| {
            let rep = c.representative;
            (rep, Transfer::new(&grid, &level, &source(), &target(rep)))
        })
        .collect();
    let mut worst = Worst::default();
    for (n, o) in offsets().into_iter().enumerate() {
        let (rep, symmetry) = class_of(o);
        let transfer = reps
            .iter()
            .find(|(r, _)| *r == rep)
            .unwrap()
            .1
            .permuted(&grid, symmetry);
        let case = Case::new(3000 + n as u64, o);
        let m = multipole(p, &level, &case);
        let reference = evaluate(p, &level, o, &rcr_local(p, &level, o, &m), &case.targets);
        let got = evaluate(
            p,
            &level,
            o,
            &diagonal_local(&grid, &level, o, &transfer, &m),
            &case.targets,
        );
        let e = bound(k, distance(o), circumradius(), p, BoundConstants::F64);
        worst.record(error(&got, &reference) / (e / THETA), || format!("{o:?}"));
    }
    worst.check(1.0, "measured / 2E over the 316 offsets");
}

/// The diagonal chain against the direct sum: within RCR's own error (its truncation)
/// plus 2E, on every class, at k a = 8 and the formula's degree at 10⁻⁶.
#[test]
fn m2l_against_the_direct_sum() {
    let ka = 8.0;
    let p = formula_degree(ka, 1e-6);
    let k = wavenumber(ka, 0.0);
    let level = Level::new(k, HALF);
    let grid = Grid::new(p);
    let mut worst_truncation: f64 = 0.0;
    for class in classes() {
        let o = class.representative;
        let case = Case::new(4000 + class.squared_distance as u64, o);
        let mut exact = Field::zero(case.targets.len());
        p2p::direct_sum(
            k,
            &case.sources,
            &case.charges,
            &case.targets,
            &mut exact.potential,
            Some(&mut exact.gradient),
        );
        let m = multipole(p, &level, &case);
        let truncation = error(
            &evaluate(p, &level, o, &rcr_local(p, &level, o, &m), &case.targets),
            &exact,
        );
        let transfer = Transfer::new(&grid, &level, &source(), &target(o));
        let got = evaluate(
            p,
            &level,
            o,
            &diagonal_local(&grid, &level, o, &transfer, &m),
            &case.targets,
        );
        let e = bound(k, distance(o), circumradius(), p, BoundConstants::F64);
        let total = error(&got, &exact);
        worst_truncation = worst_truncation.max(truncation);
        assert!(
            total <= truncation + e / THETA,
            "class {o:?}: {total:.2e} > {truncation:.2e} + 2E = {:.2e}",
            e / THETA
        );
    }
    // RCR's truncation at the formula's degree for 10⁻⁶ is of that order on these
    // placements (1.4e-6 with the corners of both boxes, the nearest class).
    assert!(worst_truncation < 1e-5, "{worst_truncation:e}");
}

/// The chain in f32 (the multipole, grid and transfer function rounded to f32, T̂ from
/// double-double), against RCR in f64, within 2E with u = 2⁻²⁴, C_T = 0.18 and C₀ = 50 at
/// k a ≈ 16 and the formula's degree at 10⁻⁴.
#[test]
fn m2l_in_f32_within_its_bound() {
    // k exact in f32, so that both chains see the same k (k a ≈ 16.02).
    let k = C::new(18.5, 0.0);
    let p = formula_degree(k.re * circumradius(), 1e-4);
    let level = Level::new(k, HALF);
    let level32 = Level::new(Complex::new(k.re as f32, k.im as f32), HALF as f32);
    let grid = Grid::<f32>::new(p);
    let from = Frame::new([0.0f32; 3], HALF as f32);
    let mut worst = Worst::default();
    for class in classes() {
        let o = class.representative;
        let to = Frame::new(o.map(|x| x as f32), HALF as f32);
        let case = Case::new(5000 + class.squared_distance as u64, o);
        let m = multipole(p, &level, &case);
        let reference = evaluate(p, &level, o, &rcr_local(p, &level, o, &m), &case.targets);
        let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
        let transfer = Transfer::new(&grid, &level32, &from, &to);
        let mut ws = diagonal::Workspace::new(p);
        let mut local32 = vec![0.0f32; m32.len()];
        diagonal::m2l(
            &grid,
            &level32,
            &from,
            &to,
            &transfer,
            &mut ws,
            &m32,
            &mut local32,
        );
        let local: Vec<f64> = local32.iter().map(|&x| x as f64).collect();
        let got = evaluate(p, &level, o, &local, &case.targets);
        let e = bound(k, distance(o), circumradius(), p, BoundConstants::F32);
        worst.record(error(&got, &reference) / (e / THETA), || format!("{o:?}"));
    }
    worst.check(1.0, "f32: measured / 2E over the 34 classes");
}

#[test]
#[should_panic(expected = "the transfer function has another degree")]
fn a_transfer_function_of_another_degree_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let grid = Grid::new(6);
    let transfer = Transfer::new(&Grid::new(5), &level, &source(), &target([2, 0, 0]));
    let m = vec![0.0; 2 * 49];
    diagonal_local(&grid, &level, [2, 0, 0], &transfer, &m);
}

#[test]
#[should_panic(expected = "the transfer function's shift")]
fn a_transfer_function_of_another_shift_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let grid = Grid::new(6);
    let transfer = Transfer::new(&grid, &level, &source(), &target([2, 0, 0]));
    let m = vec![0.0; 2 * 49];
    diagonal_local(&grid, &level, [0, 2, 0], &transfer, &m);
}

#[test]
#[should_panic(expected = "the workspace has degree 5, the grid 6")]
fn a_workspace_of_another_degree_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let grid = Grid::new(6);
    let m = vec![0.0; 2 * 49];
    let mut signature = vec![C::new(0.0, 0.0); grid.samples()];
    diagonal::s2f(
        &grid,
        &level,
        &m,
        &mut diagonal::Workspace::new(5),
        &mut signature,
    );
}

#[test]
#[should_panic(expected = "multipole has length")]
fn a_multipole_of_the_wrong_length_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let grid = Grid::new(6);
    let mut signature = vec![C::new(0.0, 0.0); grid.samples()];
    diagonal::s2f(
        &grid,
        &level,
        &[0.0; 49],
        &mut diagonal::Workspace::new(6),
        &mut signature,
    );
}

#[test]
#[should_panic(expected = "the signature has length")]
fn a_signature_of_the_wrong_length_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let grid = Grid::new(6);
    let mut local = vec![0.0; 2 * 49];
    let signature = vec![C::new(0.0, 0.0); grid.samples() - 1];
    diagonal::f2r(
        &grid,
        &level,
        &signature,
        &mut diagonal::Workspace::new(6),
        &mut local,
    );
}

#[test]
#[should_panic(expected = "the M2L needs distinct centres")]
fn a_zero_shift_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    Transfer::new(&Grid::new(4), &level, &source(), &source());
}

#[test]
#[should_panic(expected = "is not the level's half-width")]
fn a_frame_off_the_level_panics() {
    let level = Level::new(wavenumber(5.0, 0.0), HALF);
    let off = Frame::new([2.0, 0.0, 0.0], 0.25);
    Transfer::new(&Grid::new(4), &level, &source(), &off);
}

/// At the bound's floor (E ≈ C₀ u) the comparison with f64 RCR measures RCR's rounding as
/// well: on the farthest classes at k a = 54, p = 72 measured/E reaches 2.0 against RCR.
/// Against a third path, CH1.3c's `direct::m2l` (projection on a sphere), the diagonal
/// form stays within 2E and RCR errs more than it does. Release only.
#[test]
#[ignore = "large degrees: run in release by hand"]
fn at_the_floor_against_the_projection() {
    let (ka, p) = (54.0, 72);
    let k = wavenumber(ka, 0.0);
    let level = Level::new(k, HALF);
    let grid = Grid::new(p);
    for (n, o) in [
        [3, 3, 3],
        [3, 3, 2],
        [2, 3, 3],
        [2, 2, 1],
        [1, 3, 1],
        [0, 0, 2],
    ]
    .into_iter()
    .enumerate()
    {
        let case = Case::new(6000 + n as u64, o);
        let m = multipole(p, &level, &case);
        let mut ws = Workspace::new(p);
        let mut projected = vec![0.0; m.len()];
        direct::m2l(
            p,
            p,
            &level,
            &source(),
            &target(o),
            &mut ws,
            &m,
            &mut projected,
        );
        let projected = evaluate(p, &level, o, &projected, &case.targets);
        let by_rcr = evaluate(p, &level, o, &rcr_local(p, &level, o, &m), &case.targets);
        let transfer = Transfer::new(&grid, &level, &source(), &target(o));
        let by_diagonal = evaluate(
            p,
            &level,
            o,
            &diagonal_local(&grid, &level, o, &transfer, &m),
            &case.targets,
        );
        let e = bound(k, distance(o), circumradius(), p, BoundConstants::F64);
        let (diagonal, rcr) = (error(&by_diagonal, &projected), error(&by_rcr, &projected));
        println!(
            "{o:?}: diagonal {diagonal:.2e}, RCR {rcr:.2e} against the projection; against \
             each other {:.2e}; E {e:.2e}",
            error(&by_diagonal, &by_rcr)
        );
        assert!(diagonal <= e / THETA, "{o:?}: {diagonal:.2e} > 2E");
    }
}

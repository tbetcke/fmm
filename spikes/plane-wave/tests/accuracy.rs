//! The acceptance test of the spike (docs/phase6/T10-plane-wave-spike.md, "Tests that
//! define done"): the plane-wave level M2L against `nd_fmm_ref::direct::m2l` within the
//! quadrature's error.
//!
//! Error measure: the potential r φ of each local at 4 random points of every target box
//! (r = ½ box width, CONVENTIONS §3.7), |φ_pw − φ_direct| against the bound
//! E Σ_V |q|, where E is the quadrature's measured error for 1/|u| in box widths
//! (`Quadrature::measured_error`): a unit charge contributes r/|x − y| = ½ (1/|u|), so the
//! quadrature alone allows ½ E per unit charge; the bound doubles it for the dropped
//! Fourier modes and the truncation of the local at p. Direct and merge-and-shift
//! translation agree to rounding (relative 1e-12 of the largest potential).

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Frame, Workspace, direct};
use nd_fmm_spike_plane_wave::geometry::Level;
use nd_fmm_spike_plane_wave::planewave::{LevelPlan, PlaneWave, Scratch, Translation, level_m2l};
use nd_fmm_spike_plane_wave::quadrature::Quadrature;

/// A small xorshift generator; the test needs no statistics, only spread.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

const RADIUS: f64 = 0.5;

fn centre(level: &Level, b: usize) -> [f64; 3] {
    level.coords(b).map(|x| x as f64 + 0.5)
}

struct Problem {
    /// Per box: (point, charge).
    charges: Vec<Vec<([f64; 3], f64)>>,
    multipoles: Vec<f64>,
    points: Vec<Vec<[f64; 3]>>,
}

fn problem(level: Level, p: usize, seed: u64) -> Problem {
    let mut rng = Rng(seed);
    let layout = Layout::new(p);
    let n = layout.len();
    let mut multipoles = vec![0.0; level.boxes() * n];
    let mut charges = Vec::new();
    let mut points = Vec::new();
    let mut r = vec![0.0; n];
    for b in 0..level.boxes() {
        let c = centre(&level, b);
        let mut box_charges = Vec::new();
        for _ in 0..4 {
            let y: [f64; 3] = core::array::from_fn(|i| c[i] - 0.5 + rng.next());
            let q = rng.next();
            // M̃ = q conj Rₙᵐ(u), u = (y − c)/r (§3.7); conj negates the slots m < 0.
            harmonics::regular(p, core::array::from_fn(|i| (y[i] - c[i]) / RADIUS), &mut r);
            for nn in 0..=p {
                for m in -(nn as isize)..=nn as isize {
                    let idx = layout.idx(nn, m);
                    let v = if m < 0 { -r[idx] } else { r[idx] };
                    multipoles[b * n + idx] += q * v;
                }
            }
            box_charges.push((y, q));
        }
        charges.push(box_charges);
        points.push(
            (0..4)
                .map(|_| core::array::from_fn(|i| c[i] - 0.5 + rng.next()))
                .collect(),
        );
    }
    Problem {
        charges,
        multipoles,
        points,
    }
}

/// r φ of `local` (scaled) at x in box `b`.
fn potential(level: &Level, p: usize, b: usize, local: &[f64], x: [f64; 3]) -> f64 {
    let layout = Layout::new(p);
    let c = centre(level, b);
    let mut r = vec![0.0; layout.len()];
    harmonics::regular(p, core::array::from_fn(|i| (x[i] - c[i]) / RADIUS), &mut r);
    let mut sum = 0.0;
    for nn in 0..=p {
        sum += local[layout.idx(nn, 0)] * r[layout.idx(nn, 0)];
        for m in 1..=nn as isize {
            let (a, b2) = (layout.idx(nn, m), layout.idx(nn, -m));
            sum += 2.0 * (local[a] * r[a] - local[b2] * r[b2]);
        }
    }
    sum
}

struct Outcome {
    /// max over points of |φ_pw − φ_direct| / (E Σ_V |q|).
    ratio: f64,
    /// max over points of |φ_pw − φ_direct| / Σ_V |q|.
    error: f64,
    /// max |φ_merge − φ_direct-translation| / max |φ|.
    translations: f64,
    /// max |φ_direct − φ_charges| / Σ_V |q| (the M2L truncation, for context).
    truncation: f64,
}

fn check(level: Level, p: usize, q: Quadrature, seed: u64) -> Outcome {
    let e = q.measured_error();
    let prob = problem(level, p, seed);
    let n = (p + 1) * (p + 1);
    let pairs = level.pairs();
    // The reference: nd_fmm_ref's direct M2L over every V pair.
    let mut reference = vec![0.0; level.boxes() * n];
    let mut ws = Workspace::new(p);
    for pair in &pairs {
        let source = Frame::new(centre(&level, pair.source), RADIUS);
        let target = Frame::new(centre(&level, pair.target), RADIUS);
        direct::m2l(
            p,
            &source,
            &target,
            &mut ws,
            &prob.multipoles[pair.source * n..(pair.source + 1) * n],
            &mut reference[pair.target * n..(pair.target + 1) * n],
        );
    }
    let op = PlaneWave::new(p, q);
    let mut locals = Vec::new();
    for translation in [Translation::Direct, Translation::MergeShift] {
        let plan = LevelPlan::new(&op, level, translation);
        let mut scratch = Scratch::new(&op, level);
        let mut l = vec![0.0; level.boxes() * n];
        level_m2l(&op, &plan, &mut scratch, &prob.multipoles, &mut l);
        locals.push(l);
    }
    let (mut ratio, mut diff, mut largest, mut truncation): (f64, f64, f64, f64) =
        (0.0, 0.0, 0.0, 0.0);
    let mut error: f64 = 0.0;
    for t in 0..level.boxes() {
        let charge_sum: f64 = pairs
            .iter()
            .filter(|pair| pair.target == t)
            .flat_map(|pair| prob.charges[pair.source].iter())
            .map(|&(_, q)| q.abs())
            .sum();
        for &x in &prob.points[t] {
            let local = |l: &[f64]| potential(&level, p, t, &l[t * n..(t + 1) * n], x);
            let (exact_m2l, direct_pw, merged_pw) =
                (local(&reference), local(&locals[0]), local(&locals[1]));
            let charges: f64 = pairs
                .iter()
                .filter(|pair| pair.target == t)
                .flat_map(|pair| prob.charges[pair.source].iter())
                .map(|&(y, q)| {
                    let d: f64 = (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt();
                    q * RADIUS / d
                })
                .sum();
            ratio = ratio.max((merged_pw - exact_m2l).abs() / (e * charge_sum));
            ratio = ratio.max((direct_pw - exact_m2l).abs() / (e * charge_sum));
            error = error.max((merged_pw - exact_m2l).abs() / charge_sum);
            diff = diff.max((merged_pw - direct_pw).abs());
            largest = largest.max(exact_m2l.abs());
            truncation = truncation.max((exact_m2l - charges).abs() / charge_sum);
        }
    }
    Outcome {
        ratio,
        error,
        translations: diff / largest,
        truncation,
    }
}

#[test]
fn level_m2l_within_the_quadrature_error() {
    for (p, s) in [(3, 11), (6, 14), (8, 17)] {
        let q = Quadrature::with_nodes(s);
        let e = q.measured_error();
        let out = check(Level { level: 2 }, p, q, 0x5eed + p as u64);
        println!(
            "p = {p}, s = {s}: E = {e:.2e}, error / (E Σ|q|) = {:.3}, merge against direct {:.1e}, M2L truncation {:.1e} Σ|q|",
            out.ratio, out.translations, out.truncation
        );
        assert!(out.ratio <= 1.0, "p = {p}, s = {s}: {:.3}", out.ratio);
        assert!(out.translations <= 1e-12, "{:.3e}", out.translations);
    }
}

/// Level 3 and every p of the workloads: every quadrature from s = 6 until the
/// plane-wave error is at most 0.1× the M2L truncation error (the matched quadrature of
/// SPIKE_REPORT.md), and the model's choices (β = 0.1 and 1), each within the bound.
/// Release: `cargo test -p nd-fmm-spike-plane-wave --release -- --ignored --nocapture`.
#[test]
#[ignore = "release: level 3, p = 3 to 18"]
fn level_3_every_p() {
    for (p, error) in [
        (3, 2.9e-3),
        (6, 1.3e-4),
        (8, 1.9e-5),
        (12, 6.9e-7),
        (18, 1.0e-8),
    ] {
        let mut matched = None;
        for s in 6..=39 {
            let q = Quadrature::with_nodes(s);
            let e = q.measured_error();
            let out = check(Level { level: 3 }, p, q, 0xfeed + p as u64);
            println!(
                "p = {p}, s = {s}: E = {e:.2e}, error {:.2e} Σ|q| ({:.3} E), M2L truncation {:.2e} Σ|q|, merge against direct {:.1e}",
                out.error, out.ratio, out.truncation, out.translations
            );
            assert!(out.ratio <= 1.0, "p = {p}, s = {s}: {:.3}", out.ratio);
            assert!(out.translations <= 1e-12, "{:.3e}", out.translations);
            if out.error <= 0.1 * out.truncation {
                matched = Some(s);
                break;
            }
        }
        let beta: Vec<usize> = [0.1, 1.0]
            .iter()
            .map(|b| Quadrature::for_error(b * error).nodes())
            .collect();
        println!(
            "p = {p}: matched s = {matched:?}; the model's β = 0.1: s = {}, β = 1: s = {}",
            beta[0], beta[1]
        );
        assert!(matched.is_some());
    }
}

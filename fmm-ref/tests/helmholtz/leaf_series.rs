//! The series P2M → M2P and P2L → L2P against the direct sum (Helmholtz CH1.3b; proposal
//! §9 item 2, CONVENTIONS §3.15 "Radial functions and scaling").
//!
//! The box of `leaf_common` (half-width r = 0.5, circumradius a). P2M → M2P: 60 sources
//! uniform in the box and its 8 corners (ρ ≤ a), 40 targets at |v| = 2.5 a and two on the
//! z-axis. P2L → L2P: 60 sources at |ρ| = 2.5 a, 40 targets in the box, its corners, its
//! centre (where the gradient takes its limit) and two on the z-axis at |v| = a/2 (the
//! poles of the gradient's frame). Complex charges.
//!
//! Error measure: the relative ℓ2 error of proposal §9.1 over the targets, for the
//! potential and the gradient field separately, against `p2p::direct_sum`. Within the
//! truncation: at the degree p₀ of each case the error is at round-off, ≤ 1e-12 (measured
//! at most 5e-14 on the M3 Max); at a lower degree pₜ it is above 1e-7 (the series
//! converges, not trivially), and it falls from pₜ to (pₜ + p₀)/2 to p₀.

use nd_fmm_ref::helmholtz::p2p;

use crate::common::{C, SplitMix64, Worst, rel_l2, rel_l2_gradient};
use crate::leaf_common::{
    CENTRE, Field, KINDS, Kind, circumradius, inside, local_series, multipole_series, shell,
    wavenumber,
};

/// At p₀: round-off.
const CONVERGED: f64 = 1e-12;
/// At pₜ: still truncated.
const TRUNCATED: f64 = 1e-7;

/// Sources, charges, targets and the direct sum at the targets.
struct Problem {
    sources: Vec<[f64; 3]>,
    charges: Vec<C>,
    targets: Vec<[f64; 3]>,
    exact: Field,
}

impl Problem {
    fn new(k: C, sources: Vec<[f64; 3]>, charges: Vec<C>, targets: Vec<[f64; 3]>) -> Self {
        let mut exact = Field::zero(targets.len());
        p2p::direct_sum(
            k,
            &sources,
            &charges,
            &targets,
            &mut exact.potential,
            Some(&mut exact.gradient),
        );
        Self {
            sources,
            charges,
            targets,
            exact,
        }
    }

    /// The relative ℓ2 errors of a series' potential and gradient.
    fn errors(&self, field: Field) -> [f64; 2] {
        [
            rel_l2(&field.potential, &self.exact.potential),
            rel_l2_gradient(&field.gradient, &self.exact.gradient),
        ]
    }
}

/// The two targets on the z-axis through the centre at distance d (θ = 0 and π, where the
/// gradient's frame takes φ = 0).
fn on_axis(d: f64) -> [[f64; 3]; 2] {
    [d, -d].map(|z| [CENTRE[0], CENTRE[1], CENTRE[2] + z])
}

/// The two problems of one k.
struct Case {
    k: C,
    multipole: Problem,
    local: Problem,
}

impl Case {
    fn new(k: C, seed: u64) -> Self {
        let a = circumradius();
        let mut rng = SplitMix64::new(seed);
        let sources = inside(&mut rng, 60, false);
        let charges = rng.complex_charges(sources.len());
        let mut targets = shell(&mut rng, 40, 2.5 * a);
        targets.extend(on_axis(2.5 * a));
        let multipole = Problem::new(k, sources, charges, targets);
        let sources = shell(&mut rng, 60, 2.5 * a);
        let charges = rng.complex_charges(sources.len());
        let mut targets = inside(&mut rng, 40, true);
        targets.extend(on_axis(0.5 * a));
        let local = Problem::new(k, sources, charges, targets);
        Self {
            k,
            multipole,
            local,
        }
    }

    /// The errors of both series at degree p: P2M → M2P potential and gradient, then
    /// P2L → L2P potential and gradient.
    fn errors(&self, p: usize) -> [f64; 4] {
        let m = &self.multipole;
        let [a, b] = m.errors(multipole_series(
            p, self.k, &m.sources, &m.charges, &m.targets,
        ));
        let l = &self.local;
        let [c, d] = l.errors(local_series(p, self.k, &l.sources, &l.charges, &l.targets));
        [a, b, c, d]
    }
}

/// Checks one case at pₜ, (pₜ + p₀)/2 and p₀, recording the worst converged error.
fn check(kind: Kind, ka: f64, [pt, p0]: [usize; 2], worst: &mut Worst) {
    let k = wavenumber(kind, ka / circumradius());
    let case = Case::new(k, 0x3b_2026 + p0 as u64);
    let pm = (pt + p0) / 2;
    let errors = [pt, pm, p0].map(|p| case.errors(p));
    let names = ["P2M→M2P φ", "P2M→M2P ∇φ", "P2L→L2P φ", "P2L→L2P ∇φ"];
    for (j, name) in names.iter().enumerate() {
        let [truncated, middle, full] = errors.map(|e| e[j]);
        println!(
            "{kind:?} |k|a = {ka}: {name} p = {pt}: {truncated:.1e}, {pm}: {middle:.1e}, {p0}: {full:.1e}"
        );
        assert!(
            truncated > TRUNCATED,
            "{kind:?} |k|a = {ka} {name}: {truncated:.1e} at p = {pt} is not truncated"
        );
        assert!(
            truncated > middle && middle > full,
            "{kind:?} |k|a = {ka} {name}: the error does not fall"
        );
        worst.record(full, || format!("{kind:?} |k|a = {ka} {name} p = {p0}"));
    }
}

#[test]
fn series_against_the_direct_sum() {
    let mut worst = Worst::default();
    for kind in KINDS {
        for (ka, degrees) in [(0.5, [12, 40]), (5.0, [12, 48]), (30.0, [36, 76])] {
            check(kind, ka, degrees, &mut worst);
        }
    }
    worst.check(CONVERGED, "series at p₀");
}

/// Strong damping, Im(k) a = 100 (Yukawa) and about 70 (arg k = π/4): the stored
/// coefficients carry e^{∓β}, and the direct sum stays above underflow at |v| = 2.5 a.
#[test]
fn series_under_strong_damping() {
    let mut worst = Worst::default();
    for kind in [Kind::Imaginary, Kind::Complex] {
        check(kind, 100.0, [60, 116], &mut worst);
    }
    worst.check(CONVERGED, "series at p₀, strong damping");
}

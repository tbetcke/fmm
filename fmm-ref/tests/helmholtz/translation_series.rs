//! The addition theorems through RCR against the direct sum (Helmholtz CH1.3c; proposal §9
//! item 2): P2M → M2M → M2P, P2L → L2L → L2P and P2M → M2L → L2P.
//!
//! Boxes of `translation_common`: M2M from child 3 to the parent (half-width r = 0.5,
//! circumradius a_P), 40 sources uniform in the child with its corners, 40 targets at
//! 2.5 a_P from the parent's centre and two on its axis; L2L from the parent to child 4, 40
//! sources at 2.5 a_P, 40 targets in the child with its corners, its centre and its axis;
//! M2L from the box at `CENTRE` to the box at offset (3, 2, −2) (|t| ≈ 4.8 a), 40 sources
//! and 40 targets in the two boxes with their corners (the target box also its centre).
//! Complex charges; every degree p, the output's as the input's.
//!
//! Error measure: the relative ℓ2 error of proposal §9.1 over the targets, potential and
//! gradient separately, against `p2p::direct_sum`. Within the truncation, as CH1.3b's
//! series: at p₀ ≤ 1e-12, at a lower pₜ above 1e-7, falling from pₜ to (pₜ + p₀)/2 to p₀
//! until it reaches round-off (1e-13; the damped M2L at |k| a = 30 does by the middle).

use nd_fmm_ref::helmholtz::p2p;

use crate::common::{C, SplitMix64, Worst, rel_l2, rel_l2_gradient};
use crate::leaf_common::{Field, KINDS, Kind, circumradius, wavenumber};
use crate::translation_common::{Method, Translation, in_box, shell};

const CONVERGED: f64 = 1e-12;
const TRUNCATED: f64 = 1e-7;
/// Below this the error has reached round-off and need not fall further.
const AT_ROUND_OFF: f64 = 1e-13;

const CHAINS: [Translation; 3] = [
    Translation::M2m(3),
    Translation::L2l(4),
    Translation::M2l([3, 2, -2]),
];

/// One chain's points, charges and direct sum.
struct Problem {
    t: Translation,
    sources: Vec<[f64; 3]>,
    charges: Vec<C>,
    targets: Vec<[f64; 3]>,
    exact: Field,
}

impl Problem {
    fn new(t: Translation, k: C, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let (_, to) = t.frames();
        let a_out = 3.0_f64.sqrt() * to.radius;
        let sources = t.sources(&mut rng, 40);
        let charges = rng.complex_charges(sources.len());
        let axis = |d: f64| [d, -d].map(|z| [to.centre[0], to.centre[1], to.centre[2] + z]);
        let targets = match t {
            Translation::M2m(_) => {
                let mut x = shell(&mut rng, 40, &to, 2.5 * a_out);
                x.extend(axis(2.5 * a_out));
                x
            }
            Translation::L2l(_) | Translation::M2l(_) => {
                let mut x = in_box(&mut rng, 40, &to, true);
                x.extend(axis(a_out / 2.0));
                x
            }
        };
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
            t,
            sources,
            charges,
            targets,
            exact,
        }
    }

    /// The chain at degree p: the leaf operator, RCR, the evaluator.
    fn errors(&self, k: C, p: usize) -> [f64; 2] {
        let input = self.t.input(k, p, &self.sources, &self.charges);
        let output = self.t.apply(Method::Rcr, k, p, p, &input);
        let field = self.t.evaluate(k, p, &output, &self.targets);
        [
            rel_l2(&field.potential, &self.exact.potential),
            rel_l2_gradient(&field.gradient, &self.exact.gradient),
        ]
    }
}

/// One chain for one k at pₜ, (pₜ + p₀)/2 and p₀.
fn check(t: Translation, kind: Kind, ka: f64, [pt, p0]: [usize; 2], worst: &mut Worst) {
    let k = wavenumber(kind, ka / circumradius());
    let problem = Problem::new(t, k, 0x3c_2026 + p0 as u64);
    let pm = (pt + p0) / 2;
    let errors = [pt, pm, p0].map(|p| problem.errors(k, p));
    for (j, name) in ["φ", "∇φ"].iter().enumerate() {
        let [truncated, middle, full] = errors.map(|e| e[j]);
        println!(
            "{t:?} {kind:?} |k|a = {ka}: {name} p = {pt}: {truncated:.1e}, {pm}: {middle:.1e}, {p0}: {full:.1e}"
        );
        assert!(
            truncated > TRUNCATED,
            "{t:?} {kind:?} |k|a = {ka} {name}: {truncated:.1e} at p = {pt} is not truncated"
        );
        assert!(
            truncated > middle && (middle > full || middle <= AT_ROUND_OFF),
            "{t:?} {kind:?} |k|a = {ka} {name}: the error does not fall"
        );
        worst.record(full, || {
            format!("{t:?} {kind:?} |k|a = {ka} {name} p = {p0}")
        });
    }
}

/// |k| a with a the circumradius of the parent (M2M, L2L) or of the M2L's level, and the
/// degrees [pₜ, p₀]; the M2L at its offset converges faster than M2M and L2L.
fn cases(t: Translation) -> [(f64, [usize; 2]); 3] {
    match t {
        Translation::M2l(_) => [(0.5, [6, 24]), (5.0, [8, 32]), (30.0, [36, 80])],
        _ => [(0.5, [10, 40]), (5.0, [12, 48]), (30.0, [36, 80])],
    }
}

#[test]
fn addition_theorems_against_the_direct_sum() {
    let mut worst = Worst::default();
    for t in CHAINS {
        for kind in KINDS {
            for (ka, degrees) in cases(t) {
                check(t, kind, ka, degrees, &mut worst);
            }
        }
    }
    worst.check(CONVERGED, "chains at p₀");
}

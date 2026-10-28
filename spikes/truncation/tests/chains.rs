//! Test 6 of docs/helmholtz/tasks/CH0.7.md: each chain at a degree far above the
//! calibrated one against the double-double direct sum. The error left is the rounding
//! of the chain plus a truncation below it, so a bound of 10⁻²² puts the rounding at
//! least 10⁻¹² below every ε of the sweep (10⁻¹⁰ with the margin 2): what the sweep
//! measures is truncation alone.

use nd_fmm_spike_truncation::calibrate::{Chain, Config, run, worst};
use nd_fmm_spike_truncation::chain::{Kind, Setup};
use nd_fmm_spike_truncation::geometry::Placement;
use nd_fmm_spike_truncation::real::Dd;

const BOUND: f64 = 1e-22;

/// A degree at which truncation is below the bound at κ a (from the spike's own curves:
/// the low-frequency decay is about 0.16 decades per degree, the high-frequency one
/// steep beyond k a + 9 (k a)^{1/3}, and damping adds about Im(k) a), at most where the
/// scaled tables stay in range at low frequency.
fn p_big(kind: Kind, ka: f64) -> usize {
    let im = match kind {
        Kind::Real => 0.0,
        Kind::Damped(eta) => eta * ka,
        Kind::Imaginary => ka,
    };
    ((ka + 9.0 * ka.cbrt() + im).ceil() as usize + 30).max(135)
}

/// The nearest offsets on and off the axes (β = 0, π and generic), and the farthest.
const OFFSETS: [[i32; 3]; 5] = [[2, 0, 0], [0, 0, -2], [0, 0, 2], [2, 1, -1], [3, 3, 3]];

fn check(kind: Kind, ka: f64, chains: &[Chain]) {
    let setup = Setup::<Dd>::new(kind, ka);
    let deg = [
        p_big(kind, ka),
        p_big(kind, ka / 2.0),
        p_big(kind, ka / 4.0),
    ];
    let cfg = Config::default();
    for &chain in chains {
        let rec = run(&setup, chain, &OFFSETS, &Placement::ALL, &[deg], &cfg);
        assert_eq!(rec.len(), OFFSETS.len() * 3);
        let (pot, grad) = worst(&rec, 1)[0];
        println!(
            "{} {} ka {ka} p {deg:?}: pot {pot:.2e} grad {grad:.2e}",
            chain.name(),
            kind.name()
        );
        assert!(
            pot < BOUND && grad < BOUND,
            "{} {} ka {ka}: {pot:e} {grad:e}",
            chain.name(),
            kind.name()
        );
    }
}

const ALL: [Chain; 4] = [Chain::C1, Chain::C2, Chain::C3m, Chain::C3l];

#[test]
fn only_truncation_low_frequency() {
    check(Kind::Real, 4.0, &ALL);
}

#[test]
#[ignore = "every kind at k a = 0.1, 4, 32; run in release"]
fn only_truncation_every_kind() {
    for kind in Kind::ALL {
        for ka in [0.1, 4.0, 32.0] {
            check(kind, ka, &ALL);
        }
    }
}

#[test]
#[ignore = "degrees to 343 in double-double (about 15 minutes); run in release"]
fn only_truncation_high_frequency() {
    check(Kind::Real, 128.0, &ALL);
    check(Kind::Damped(0.1), 128.0, &ALL);
    check(Kind::Real, 256.0, &ALL);
}

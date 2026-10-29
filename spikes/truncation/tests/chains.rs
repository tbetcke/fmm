//! Test 6 of docs/helmholtz/tasks/CH0.7.md: each chain at a degree far above the
//! calibrated one against the double-double direct sum. The error left is the rounding
//! of the chain plus a truncation below it, so a bound of 10⁻²⁰ puts the rounding at
//! least 5·10⁹ below every ε/2 of the sweep (5·10⁻¹¹): what the sweep measures is
//! truncation alone. (At low frequency the decay is slow, about 0.16 decades per degree,
//! slower with damping, and the scaled tables end near p = 135, where the gradient of C2
//! reaches 10⁻²² for real k and 5·10⁻²¹ for η = 1.) Every case with Im(k) a ≥ 16 has the
//! larger bound of [`bound`].

use nd_fmm_spike_truncation::calibrate::{Chain, Config, run, worst};
use nd_fmm_spike_truncation::chain::{Kind, Setup};
use nd_fmm_spike_truncation::geometry::Placement;
use nd_fmm_spike_truncation::real::Dd;

/// Im(k) a.
fn im_ka(kind: Kind, ka: f64) -> f64 {
    match kind {
        Kind::Real => 0.0,
        Kind::Damped(eta) => eta * ka,
        Kind::Imaginary => ka,
    }
}

/// A degree at which truncation is below the bound at κ a (from the spike's own curves:
/// the low-frequency decay is about 0.16 decades per degree, the high-frequency one
/// steep beyond k a + 9 (k a)^{1/3}, and damping adds about Im(k) a, 3.5 Im(k) a when
/// strong), at most where the scaled tables stay in range at low frequency.
fn p_big(kind: Kind, ka: f64) -> usize {
    let strong = matches!(kind, Kind::Imaginary) || matches!(kind, Kind::Damped(eta) if eta >= 1.0);
    let im = im_ka(kind, ka) * if strong { 3.5 } else { 1.0 };
    // At k a ≥ 16 the scale is 1 and the tables stay in range beyond 135.
    let floor = if ka >= 16.0 { 160 } else { 135 };
    ((ka + 9.0 * ka.cbrt() + im).ceil() as usize + 30).max(floor)
}

/// The bound: 10⁻²⁰, or 10⁻¹⁶ where Im(k) a ≥ 16. There the damped field across an
/// offset is decades below its terms, and the double-double chain levels off at up to
/// 6·10⁻¹⁷ (η = 1 and imaginary k at κ a = 32; η = 0.1 at k a = 256, where the shift
/// recursion and the damped M2L projection set it), still 10⁶ below ε/2.
fn bound(kind: Kind, ka: f64) -> f64 {
    if im_ka(kind, ka) >= 16.0 {
        1e-16
    } else {
        1e-20
    }
}

/// The nearest offsets on and off the axes (β = 0, π and generic), and the farthest.
const OFFSETS: [[i32; 3]; 5] = [[2, 0, 0], [0, 0, -2], [0, 0, 2], [2, 1, -1], [3, 3, 3]];

fn check(kind: Kind, ka: f64, chains: &[Chain]) {
    let bound = bound(kind, ka);
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
            pot < bound && grad < bound,
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
#[ignore = "degrees to 369 in double-double (about 20 minutes); run in release"]
fn only_truncation_high_frequency() {
    check(Kind::Real, 128.0, &ALL);
    check(Kind::Damped(0.1), 128.0, &ALL);
    check(Kind::Real, 256.0, &ALL);
    check(Kind::Damped(0.1), 256.0, &ALL);
}

/// The record behind SPIKE_REPORT.md §2.4: S2's outer-sphere projection of the shift
/// tables, accurate in M2M's weights, fails in L2L with complex k. C3l with η = 0.1 at
/// k a = 128, at the degrees of `check` and at p = 220 on every level: the recursion
/// leaves at most 10⁻²⁰; the projection 1.4·10⁻¹⁰ (above ε/2 at ε = 10⁻¹⁰) and 6·10¹⁹.
#[test]
#[ignore = "projection tables to p = 220 in double-double; run in release"]
fn l2l_by_projection_fails_with_damping() {
    let kind = Kind::Damped(0.1);
    let ka = 128.0;
    let cfg = Config::default();
    let realistic = [
        p_big(kind, ka),
        p_big(kind, ka / 2.0),
        p_big(kind, ka / 4.0),
    ];
    for (deg, at_least) in [(realistic, 1e-11), ([220; 3], 1.0)] {
        let mut errors = Vec::new();
        for projection in [false, true] {
            let mut setup = Setup::<Dd>::new(kind, ka);
            setup.shift_projection = Some(projection);
            let rec = run(&setup, Chain::C3l, &OFFSETS, &Placement::ALL, &[deg], &cfg);
            let (pot, grad) = worst(&rec, 1)[0];
            println!(
                "C3l η = 0.1, k a = 128, p {deg:?}, projection {projection}: pot {pot:.2e} grad {grad:.2e}"
            );
            errors.push(pot.max(grad));
        }
        assert!(errors[0] < 1e-20, "recursion {:e}", errors[0]);
        assert!(errors[1] > at_least, "projection {:e}", errors[1]);
    }
}

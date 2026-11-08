//! The α counts of the exponential M2L at the largest V offsets (Helmholtz CH6.2g; CONVENTIONS
//! §3.15, "Exponential M2L"): two points of the 24 offsets whose components are a permutation
//! of (±3, ±3, ±2) lie up to the region's corner 4√2 apart laterally, which the default counts
//! (`Quadrature::new`, reach `RHO_MAX`) bound. These tests check the FMM's chain there: a
//! multipole of degree p_l about the source box's centre, the pair translation along each
//! direction merge and shift gives the offset (`PlaneWaveLevel::m2l`, which merge and shift
//! equals to 1e-12 relative, `nd-fmm-tables`' `tests/helmholtz/plane_wave_merge.rs`), a local
//! of degree p_l, at the level's degree of the calibrated rule (`degree`), with iprec 2 (the
//! rule's precision).
//!
//! Error measure (`nd-fmm-tables`' `tests/helmholtz/plane_wave_pair.rs`, Helmholtz CH6.1c):
//! the multipole of point charges at the source box's corners, its centre and four seeded
//! points (Σ|q| = 1, P2M in stored form) is translated to each largest offset by RCR
//! (`HelmholtzTables::m2l`, Rotation's pair translation) and by the exponential M2L, and both
//! locals are evaluated by L2P at the 27 points {−½, 0, ½}³ (box widths) of the target box,
//! the corners included; the error is the largest |Δ| in box widths (twice the level units' r_l
//! φ̃, the quadrature's norm). At the level's degree the chain also errs on the truncated
//! multipole's tail (CH6.1c's degree study), which the counts do not govern, so the error is
//! compared with the error of the same nodes with **converged** counts,
//! Nₖ = ⌈|tₖ| 4√2⌉ + 2Fₖ + 40, far beyond any aliasing: the default counts may add at most
//! [`COUNTS_ADDED`] times the stated error to it. That bound is an empirical regression guard,
//! not a derived one: the measured worst is 3.2× (damped domains), and tighter bounds (1.1×
//! the converged counts' error, with or without the stated error added) failed on damped
//! domains where the counts do add a little; it catches aliasing of the kind a smaller reach
//! causes at high κ, not a regression of a few times the stated error. The counts for reach
//! 4.5 (CH6.2e's proposal) are measured beside and printed, not asserted: on the weakly damped
//! domains (Im κ < π/2) they add more than 10× the stated error from Re κ = 7π (16× there at
//! ε = 10⁻⁶, 47× from 8π, 2·10⁴× near 16π), and on some damped domains from 1.5π (up to 16×),
//! which is why the default keeps the corner (fmm-validate/results/helmholtz-ch6-2g.md §2).
//! The domains from Im κ = [`DAMPED`] are left out, a known effect deferred to Helmholtz
//! CH6.2f: the field across a V offset is below e^{−2·5π} ≈ 2e-14 of the near field there and
//! the stated errors fall from 6·10⁻⁸ to 4·10⁻¹⁶, so a bound in their units means nothing,
//! while both counts' chains err by 10⁻⁹ to 10⁻⁵ absolute, the default's up to about 3× the
//! converged counts' (the report's §3).

use nd_fmm_exec::helmholtz::{Wavenumber, degree};
use nd_fmm_math::Layout;
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_fmm_tables::helmholtz::plane_wave::pair::{Nodes, PlaneWaveLevel};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{DOMAINS, RHO_MAX, rectangle};
use nd_fmm_tables::helmholtz::plane_wave::{Precision, Quadrature};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use nd_fmm_tables::plane_wave::geometry::{Direction, slot_of};

use super::common::{C, Rng};

/// The most the default counts may add to the converged counts' error, in units of the stated
/// error (module documentation).
const COUNTS_ADDED: f64 = 10.0;

/// The domains from this Im κ on are left out (module documentation): 5π, a breakpoint of
/// FMM3D's tables.
const DAMPED: f64 = 5.0 * core::f64::consts::PI;

/// The reach CH6.2e proposed, measured beside the default.
const PROPOSED_REACH: f64 = 4.5;

/// The level the pairs are on: κ = 2k̂₀/2² there.
const LEVEL: usize = 2;

/// The circumradius of a box of width 1, the degree rule's a.
const CIRCUMRADIUS: f64 = 0.866_025_403_784_438_6;

/// The indices of the 24 largest V offsets, the permutations of (±3, ±3, ±2).
fn largest_offsets() -> Vec<usize> {
    let offsets = m2l_offsets();
    let largest: Vec<usize> = (0..offsets.len())
        .filter(|&i| {
            let mut d = offsets[i].map(i64::abs);
            d.sort_unstable();
            d == [2, 3, 3]
        })
        .collect();
    assert_eq!(largest.len(), 24);
    largest
}

/// The directions of the pair at the offset index `index`: [`Direction::of`] its offset and
/// those that merge and shift gives it at any child ([`Direction::for_slot`]).
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

/// The sources (level units, half-width 1, the source box at the origin): the corners, the
/// centre and four seeded points; seeded charges, real if `real`, with Σ|q| = 1.
fn sources(real: bool) -> (Vec<[f64; 3]>, Vec<C>) {
    let mut rng = Rng(0xc462_0e5a);
    let mut points: Vec<[f64; 3]> = (0..8)
        .map(|c| [c & 4, c & 2, c & 1].map(|b| if b == 0 { -1.0 } else { 1.0 }))
        .collect();
    points.push([0.0; 3]);
    for _ in 0..4 {
        points.push([(); 3].map(|_| rng.next()));
    }
    let mut charges: Vec<C> = points
        .iter()
        .map(|_| {
            let im = if real { 0.0 } else { 0.5 * rng.next() };
            C::new(0.5 * rng.next() + 0.75, im)
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

/// The largest error (the module documentation) of the exponential M2L of each level in
/// `waves` over the largest offsets and their directions, with `planes` multipole planes
/// (1: real charges, real k).
fn errors(rcr: &HelmholtzTables<f64>, waves: &[PlaneWaveLevel<f64>], planes: usize) -> Vec<f64> {
    let p = rcr.degree(LEVEL);
    let len = Layout::new(p).len();
    let k_level = Level::new(rcr.k_hat(LEVEL), 1.0);
    let mut ws = Workspace::new(p);
    let (points, charges) = sources(planes == 1);
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
    let mut worst = vec![0.0f64; waves.len()];
    for index in largest_offsets() {
        let d = m2l_offsets()[index];
        let to = Frame::new(d.map(|x| 2.0 * x as f64), 1.0);
        let targets = targets(d);
        let mut expected = vec![0.0; 2 * len];
        rcr.m2l(
            LEVEL,
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
        for (level, worst) in waves.iter().zip(&mut worst) {
            let mut scratch = level.scratch();
            for direction in directions(index) {
                let mut local = vec![0.0; 2 * len];
                level.m2l(
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
                    *worst = worst.max(2.0 * (*a - *b).norm());
                }
            }
        }
    }
    worst
}

/// The level of `q`'s nodes with converged α counts (the module documentation).
fn converged(k_hat: C, p: usize, q: &Quadrature) -> PlaneWaveLevel<f64> {
    let alpha: Vec<usize> = q
        .t()
        .iter()
        .zip(q.fourier_counts())
        .map(|(t, &f)| (t.norm() * RHO_MAX).ceil() as usize + 2 * f + 40)
        .collect();
    let nodes = Nodes {
        lambda: q.lambda(),
        t: q.t(),
        weights: q.weights(),
        alpha: &alpha,
        fourier: q.fourier_counts(),
    };
    PlaneWaveLevel::from_nodes(k_hat, p, nodes)
}

/// The largest additions to the converged counts' error seen, over the stated error: the
/// default counts' and reach 4.5's, with the κ of the latter.
struct Worst {
    default: f64,
    proposed: (f64, C),
}

impl Worst {
    fn new() -> Self {
        Self {
            default: 0.0,
            proposed: (0.0, C::new(0.0, 0.0)),
        }
    }
}

/// Checks the largest offsets at κ (box widths) and ε: the default counts' error against RCR
/// within the converged counts' plus [`COUNTS_ADDED`] times the stated error; reach 4.5's
/// recorded.
fn check(kappa: C, epsilon: f64, worst: &mut Worst) {
    let k = Wavenumber::new(kappa.re, kappa.im).expect("a valid wavenumber");
    let p = degree(&k, CIRCUMRADIUS, epsilon);
    let rcr = HelmholtzTables::<f64>::build(kappa * 2.0, &[0, 0, p], DEFAULT_BLOCK_BUDGET);
    let k_hat = rcr.k_hat(LEVEL);
    let default = Quadrature::new(kappa, Precision::Iprec2).expect("κ in the tables");
    assert_eq!(default.reach(), RHO_MAX);
    let proposed = Quadrature::with_reach(kappa, Precision::Iprec2, PROPOSED_REACH).unwrap();
    let waves = [
        PlaneWaveLevel::<f64>::new(k_hat, p, &default),
        converged(k_hat, p, &default),
        PlaneWaveLevel::<f64>::new(k_hat, p, &proposed),
    ];
    let planes: &[usize] = if rcr.real_shifts() { &[1, 2] } else { &[2] };
    for &planes in planes {
        let errors = errors(&rcr, &waves, planes);
        let (ours, best, smaller) = (errors[0], errors[1], errors[2]);
        let e = default.stated_error();
        let added = |error: f64| (error - best) / e;
        worst.default = worst.default.max(added(ours));
        if added(smaller) > worst.proposed.0 {
            worst.proposed = (added(smaller), kappa);
        }
        assert!(
            ours <= best + COUNTS_ADDED * e,
            "κ = {kappa:.4}, domain {}, ε = {epsilon:.0e}, p = {p}, {planes} plane(s): the \
             default counts {ours:.3e}, converged {best:.3e} ({:.3}×; stated {e:.1e})",
            default.domain(),
            ours / best,
        );
    }
}

/// The points of a domain checked: its centre, and on the domains of real κ's edge
/// (Im κ = 0) the centre's real part too, where real charges give one multipole plane; none
/// from Im κ = [`DAMPED`] on.
fn points(domain: usize) -> Vec<C> {
    let (lo, hi) = rectangle(domain);
    if lo.im >= DAMPED * (1.0 - 1e-12) {
        return Vec::new();
    }
    let centre = (lo + hi) * 0.5;
    let mut out = vec![centre];
    if lo.im == 0.0 {
        out.push(C::new(centre.re, 0.0));
    }
    out
}

/// In debug builds: the domains of CH6.1a's fixture (1: κ ≈ 0; 11: Re κ in [3π, 4π]; 139:
/// Im κ in [1, π/2]; 172: κ near 10 + 3i; 236: κ near 0.5 + 10i) and domain 92 (Re κ in
/// [15π, 16π], Im κ in [0.1, 0.2], where reach 4.5 errs most), at ε = 10⁻⁶.
#[test]
fn the_counts_at_the_largest_offsets() {
    let mut worst = Worst::new();
    for domain in [1, 11, 139, 172, 236, 92] {
        for kappa in points(domain) {
            check(kappa, 1e-6, &mut worst);
        }
    }
    println!(
        "ε = 1e-6: the default counts add at most {:.3} × the stated error to the converged \
         counts' error, reach {PROPOSED_REACH} {:.3e} × (κ = {:.3})",
        worst.default, worst.proposed.0, worst.proposed.1
    );
    assert!(
        worst.proposed.0 > 100.0,
        "reach {PROPOSED_REACH} adds orders of magnitude more than the stated error on domain 92"
    );
}

/// Every domain of iprec 2 below Im κ = [`DAMPED`] at ε = 10⁻³ and 10⁻⁶, the ends of iprec
/// 2's range under the rule (release, about 40 s on the M6).
#[test]
#[ignore = "release: 437 domains at two tolerances"]
fn the_counts_at_the_largest_offsets_on_every_domain() {
    for epsilon in [1e-3, 1e-6] {
        let mut worst = Worst::new();
        for domain in 1..=DOMAINS {
            for kappa in points(domain) {
                check(kappa, epsilon, &mut worst);
            }
        }
        println!(
            "every domain, ε = {epsilon:.0e}: the default counts add at most {:.3} × the stated \
             error to the converged counts' error, reach {PROPOSED_REACH} {:.3e} × (κ = {:.3})",
            worst.default, worst.proposed.0, worst.proposed.1
        );
    }
}

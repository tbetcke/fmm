//! The degree rules: S7's closed form (Helmholtz CH3.1a; spike S7,
//! `spikes/truncation/results/summary.md`, "Per-level degree p*") at its grid points, never
//! below p*; the calibrated default (Helmholtz CH3.3, `fmm-validate/results/helmholtz-ch3-3.md`)
//! never above S7's, monotone, at its fitted points; and `Degrees::resolve`. Exact integers.

use nd_fmm_exec::geometry::{Domain, radius};
use nd_fmm_exec::helmholtz::{DegreeError, DegreeRule, Degrees, Wavenumber, degree, s7_degree};
use nd_octree::PhysicalBox;

const EPSILONS: [f64; 8] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];

/// Rows of S7's real-k table at k a = 2^j: (k a, [(p*, closed form) per ε]).
const ROWS: [(f64, [(usize, usize); 8]); 7] = [
    (
        1.0,
        [
            (12, 16),
            (17, 22),
            (23, 28),
            (29, 34),
            (35, 40),
            (42, 46),
            (49, 52),
            (54, 58),
        ],
    ),
    (
        4.0,
        [
            (11, 16),
            (15, 22),
            (20, 28),
            (26, 34),
            (31, 40),
            (38, 46),
            (44, 52),
            (50, 58),
        ],
    ),
    (
        16.0,
        [
            (24, 26),
            (26, 28),
            (28, 29),
            (30, 34),
            (33, 40),
            (38, 46),
            (44, 52),
            (50, 58),
        ],
    ),
    (
        32.0,
        [
            (41, 44),
            (44, 46),
            (46, 49),
            (49, 51),
            (51, 53),
            (53, 55),
            (55, 56),
            (57, 58),
        ],
    ),
    (
        64.0,
        [
            (75, 79),
            (79, 82),
            (82, 85),
            (85, 88),
            (88, 90),
            (90, 92),
            (93, 95),
            (95, 97),
        ],
    ),
    (
        128.0,
        [
            (141, 147),
            (146, 151),
            (150, 154),
            (153, 158),
            (157, 161),
            (161, 164),
            (164, 167),
            (167, 169),
        ],
    ),
    (
        256.0,
        [
            (272, 280),
            (278, 284),
            (284, 289),
            (288, 293),
            (293, 297),
            (297, 301),
            (300, 304),
            (304, 308),
        ],
    ),
];

#[test]
fn the_closed_form_at_s7_grid_points() {
    for (ka, row) in ROWS {
        let k = Wavenumber::new(ka, 0.0).unwrap();
        for (epsilon, (p_star, closed)) in EPSILONS.into_iter().zip(row) {
            let p = s7_degree(&k, 1.0, epsilon);
            assert_eq!(p, closed, "k a = {ka}, ε = {epsilon:e}");
            assert!(
                p >= p_star,
                "k a = {ka}, ε = {epsilon:e}: {p} < p* = {p_star}"
            );
        }
    }
}

/// The largest measured degree at ε/2 per ε (10⁻³ … 10⁻⁸) at the levels' x of CH3.3's
/// scans (`fmm-validate/results/helmholtz-ch3-3.md`): (x, [p per ε]), the plateau first at
/// x = 0 (the envelope of the plateau and low-x level scans, x ≤ 4).
const MEASURED: [(f64, [usize; 6]); 1] = [(0.0, [7, 9, 12, 15, 20, 24])];

#[test]
fn the_calibrated_rule_covers_the_measured_degrees() {
    for (x, row) in MEASURED {
        for (epsilon, p) in EPSILONS.into_iter().zip(row) {
            let rule = DegreeRule::CALIBRATED.degree(x, epsilon);
            assert!(rule >= p, "x = {x}, ε = {epsilon:e}: {rule} < {p}");
        }
    }
}

#[test]
fn the_calibrated_rule_is_at_most_s7s_and_monotone() {
    let xs: Vec<f64> = (-30..=33).map(|j| 2f64.powf(j as f64 / 4.0)).collect();
    let epsilons: Vec<f64> = (30..=100).map(|j| 10f64.powf(-j as f64 / 10.0)).collect();
    for &x in &xs {
        let k = Wavenumber::new(x, 0.0).unwrap();
        for (i, &epsilon) in epsilons.iter().enumerate() {
            let p = degree(&k, 1.0, epsilon);
            assert_eq!(p, DegreeRule::CALIBRATED.degree(x, epsilon));
            assert!(
                p <= s7_degree(&k, 1.0, epsilon),
                "x = {x}, ε = {epsilon:e}: {p} above S7's"
            );
            if i > 0 {
                assert!(p >= degree(&k, 1.0, epsilons[i - 1]), "not monotone in ε");
            }
            assert!(p >= degree(&k, 0.9, epsilon), "not monotone in x");
        }
    }
}

#[test]
fn degree_rule_shapes() {
    // S7's closed form is γ = 0, α = 6, β = −2, C = 1 (up to the forgiving ceilings).
    let s7 = DegreeRule {
        gamma: 0.0,
        alpha: 6.0,
        beta: -2.0,
        c: 1.0,
    };
    for (ka, row) in ROWS {
        for (epsilon, (_, closed)) in EPSILONS.into_iter().zip(row) {
            assert_eq!(
                s7.degree(ka, epsilon),
                closed,
                "k a = {ka}, ε = {epsilon:e}"
            );
        }
    }
    // The ceilings forgive 10⁻⁹: 2.5 · 6 + 1 = 16 exactly at ε = 10⁻⁶.
    let rule = DegreeRule {
        gamma: 0.0,
        alpha: 2.5,
        beta: 1.0,
        c: 0.0,
    };
    assert_eq!(rule.plateau(1e-6), 16);
    assert_eq!(rule.high(0.0, 1e-6), 0);
    assert_eq!(rule.degree(20.0, 1e-6), 20);
}

#[test]
fn complex_k_takes_its_real_part() {
    let real = Wavenumber::new(30.0, 0.0).unwrap();
    let complex = Wavenumber::new(30.0, 3.0).unwrap();
    for epsilon in EPSILONS {
        assert_eq!(degree(&complex, 0.7, epsilon), degree(&real, 0.7, epsilon));
    }
}

#[test]
fn resolve() {
    let domain = Domain::new(&PhysicalBox::new([-1.0, -1.0, -1.0, 3.0, 3.0, 3.0])).unwrap();
    let k = Wavenumber::new(50.0, 0.0).unwrap();
    // Side 4: r₂ = 0.5, x₂ = 50 · 0.866 = 43.3 and ⌈43.3 + 0.8 · 18.42^{2/3} · 43.3^{1/3}⌉
    // = ⌈43.3 + 0.8 · 6.975 · 3.512⌉ = 63; level 3 ⌈21.65 + 0.8 · 6.975 · 2.787⌉ = 38;
    // level 4 (x = 10.8) and 5 the plateau ⌈0.30 · 64 + 0.25 · 8 + 2.60⌉ = 24.
    let degrees = Degrees::Tolerance(1e-8).resolve(&k, &domain, 6).unwrap();
    assert_eq!(degrees, [0, 0, 63, 38, 24, 24]);
    // S7's closed form: 68 on level 2, its floor ⌈6 · 8⌉ − 2 = 46 below.
    let s7: Vec<usize> = (2..6)
        .map(|l| s7_degree(&k, 3f64.sqrt() * radius(l, &domain), 1e-8))
        .collect();
    assert_eq!(s7, [68, 46, 46, 46]);

    let named = Degrees::PerLevel(vec![1, 2, 3, 4, 5]);
    assert_eq!(named.resolve(&k, &domain, 3).unwrap(), [1, 2, 3]);
    assert_eq!(
        named.resolve(&k, &domain, 6),
        Err(DegreeError::TooFewLevels {
            given: 5,
            levels: 6
        })
    );
    for epsilon in [0.0, 1.0, -1e-6, f64::NAN] {
        assert!(matches!(
            Degrees::Tolerance(epsilon).resolve(&k, &domain, 3),
            Err(DegreeError::Tolerance(_))
        ));
    }
}

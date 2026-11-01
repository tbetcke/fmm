//! The degree rule (Helmholtz CH3.1a; spike S7, `spikes/truncation/results/summary.md`,
//! "Per-level degree p*"): S7's closed form at its grid points, never below p*, and
//! `Degrees::resolve`. Exact integers.

use nd_fmm_exec::geometry::Domain;
use nd_fmm_exec::helmholtz::{DegreeError, Degrees, Wavenumber, degree};
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
            let p = degree(&k, 1.0, epsilon);
            assert_eq!(p, closed, "k a = {ka}, ε = {epsilon:e}");
            assert!(
                p >= p_star,
                "k a = {ka}, ε = {epsilon:e}: {p} < p* = {p_star}"
            );
        }
    }
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
    // Side 4: r₂ = 0.5, k a₂ = 50 · 0.866 = 43.3 and ⌈43.3 + 18.42^{2/3} · 43.3^{1/3}⌉ =
    // ⌈43.3 + 6.975 · 3.512⌉ = 68; level 3 (21.7 + 19.4 = 41.1) and finer the floor
    // ⌈6 · 8⌉ − 2 = 46.
    let degrees = Degrees::Tolerance(1e-8).resolve(&k, &domain, 6).unwrap();
    assert_eq!(degrees, [0, 0, 68, 46, 46, 46]);

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

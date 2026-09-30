//! Level independence (C2.3): the tables built at the canonical frames reproduce
//! `nd_fmm_ref::rotation` at the actual frames of V-list pairs and parent–child pairs
//! on levels 2, 9 and 16 of a dyadic domain.

use nd_fmm_ref::Workspace;
use nd_fmm_tables::rotation::Operator;
use nd_fmm_tables::{MatrixSet, RotationTables};

use crate::common::{
    DEBUG_DEGREES, LEVELS, OPERATORS, REFERENCE_TOL, SplitMix64, Worst, dense, error, input_kind,
    level_frames, random_coefficients, reference, run, tables, translate,
};

/// Entry `t` of `op` of `rotation` against `nd_fmm_ref::rotation` at random frames of
/// that entry on `level` of the dyadic domain ([`level_frames`]), for a random input.
/// Returns the per-degree error relative to the terms of the dense matrix `t` of `set`.
pub fn check_level(
    rotation: &RotationTables<f64>,
    set: &MatrixSet<f64>,
    op: Operator,
    level: u32,
    t: usize,
    ws: &mut Workspace<f64>,
    rng: &mut SplitMix64,
) -> f64 {
    let p = rotation.p();
    let (from, to) = level_frames(op, level, t, rng);
    let x = random_coefficients(input_kind(op), p, rng);
    let got = run(rotation.tables(op), t, &x);
    let want = translate(reference(op), p, (&from, &to), ws, &x);
    error(op, set, t, &x, &got, &want)
}

/// Runs [`check_level`] for every entry of every family on every level of `LEVELS`,
/// `pairs` times each, and asserts `REFERENCE_TOL`.
pub fn check_levels(
    rotation: &RotationTables<f64>,
    set: impl Fn(Operator) -> &'static MatrixSet<f64>,
    pairs: usize,
    worst: &[Worst; 3],
    rng: &mut SplitMix64,
) {
    let p = rotation.p();
    let mut ws = Workspace::new(p);
    for level in LEVELS {
        for (op, worst) in OPERATORS.into_iter().zip(worst) {
            for t in 0..op.count() {
                for _ in 0..pairs {
                    let e = check_level(rotation, set(op), op, level, t, &mut ws, rng);
                    worst.update(e);
                    assert!(
                        e <= REFERENCE_TOL,
                        "{op:?}, level {level}, entry {t}, p = {p}: {e:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn operators_equal_reference_rotation_on_levels_2_9_16_of_a_dyadic_domain() {
    // Error measure: output coefficients per degree in the orthonormal weighting,
    // relative to the terms |Aᵢₖ xₖ| of the dense table (`main`). The dyadic domain
    // a = (−1.25, 0.5, 2), w = 3 makes every centre and centre difference exact, so
    // `nd_fmm_ref::rotation` sees the canonical scaled shift and radius ratio exactly
    // (asserted by `level_frames`); its angles and blocks come from the shift on the
    // level. Levels are those of the V-list pair for M2L and of the child for M2M and
    // L2L. All 316 offsets and all 8 octants, two random pairs per entry and level
    // (four for the octants), p ≤ 8.
    let mut rng = SplitMix64::new(0x0706_0201);
    let worst = [
        Worst::new("M2L tables vs rotation, levels 2, 9, 16, p ≤ 8 (terms)"),
        Worst::new("M2M tables vs rotation, levels 2, 9, 16, p ≤ 8 (terms)"),
        Worst::new("L2L tables vs rotation, levels 2, 9, 16, p ≤ 8 (terms)"),
    ];
    for p in DEBUG_DEGREES {
        check_levels(tables(p), |op| dense(p).of(op), 2, &worst, &mut rng);
        // More pairs for the 8 octants: they are cheap.
        let mut ws = Workspace::new(p);
        for level in LEVELS {
            for (op, worst) in [Operator::M2m, Operator::L2l].into_iter().zip(&worst[1..]) {
                for t in 0..op.count() {
                    for _ in 0..2 {
                        let set = dense(p).of(op);
                        let e = check_level(tables(p), set, op, level, t, &mut ws, &mut rng);
                        worst.update(e);
                        assert!(
                            e <= REFERENCE_TOL,
                            "{op:?}, level {level}, o = {t}: {e:.3e}"
                        );
                    }
                }
            }
        }
    }
}

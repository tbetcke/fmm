//! The C2.3 criterion as the Phase 1 gate measured it: the table-driven rotation
//! operators against `direct`, through the dense tables of T3 and T4 and against
//! `nd_fmm_ref::direct` itself, at the canonical frames.

use nd_fmm_ref::Workspace;

use crate::common::{
    DEBUG_DEGREES, DIRECT_TOL, OPERATORS, SplitMix64, Worst, apply, dense, direct_operator, error,
    input_kind, random_coefficients, run, tables, translate,
};

#[test]
fn operators_equal_direct_through_the_dense_tables() {
    // Error measure: output coefficients per degree in the orthonormal weighting,
    // relative to the terms |Aᵢₖ xₖ| of the dense table (`main`), bound 1e-13. The
    // dense tables of T3 and T4 are `direct` at the canonical frames, column by column;
    // the second comparison calls `direct` itself. All 316 offsets and all 8 octants,
    // two random inputs each, p ≤ 8.
    let mut rng = SplitMix64::new(0x0706_0101);
    let worst = [
        Worst::new("M2L tables vs direct (dense and direct), p ≤ 8 (terms)"),
        Worst::new("M2M tables vs direct (dense and direct), p ≤ 8 (terms)"),
        Worst::new("L2L tables vs direct (dense and direct), p ≤ 8 (terms)"),
    ];
    for p in DEBUG_DEGREES {
        let mut ws = Workspace::new(p);
        for (op, worst) in OPERATORS.into_iter().zip(&worst) {
            let family = tables(p).tables(op);
            let set = dense(p).of(op);
            for t in 0..op.count() {
                let (from, to) = op.frames(t);
                for _ in 0..2 {
                    let x = random_coefficients(input_kind(op), p, &mut rng);
                    let got = run(family, t, &x);
                    for want in [
                        apply(set, t, &x),
                        translate(direct_operator(op), p, (&from, &to), &mut ws, &x),
                    ] {
                        let e = worst.update(error(op, set, t, &x, &got, &want));
                        assert!(e <= DIRECT_TOL, "{op:?}, p = {p}, entry {t}: {e:.3e}");
                    }
                }
            }
        }
    }
}

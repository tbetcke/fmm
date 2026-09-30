//! f32 tables: the f64 tables rounded entry by entry, applied in f32, against the f64
//! tables applied in f64.

use nd_fmm_tables::RotationTables;
use nd_fmm_tables::rotation::RotationScratch;

use crate::common::{
    OPERATORS, P_MAX_F32, SplitMix64, Worst, dense, error, input_kind, len, random_coefficients,
    run, tables,
};

/// Tolerance of the f32 tables, per degree relative to the terms (brief T6).
const F32_TOL: f64 = 1e-5;

#[test]
fn f32_tables_are_the_f64_tables_rounded() {
    // Error measure: bit-for-bit equality of the f32 build with the f64 build cast to
    // f32.
    for p in [0, 4, P_MAX_F32] {
        assert_eq!(
            RotationTables::<f32>::build(p),
            tables(p).cast::<f32>(),
            "p = {p}"
        );
    }
}

#[test]
fn f32_tables_match_f64_to_single_precision() {
    // The input is rounded to f32 first, so both runs see the same coefficients.
    //
    // Error measure: output coefficients per degree in the orthonormal weighting,
    // relative to the terms |Aᵢₖ xₖ| of the dense f64 table (`main`). All entries, two
    // random inputs each, p ≤ 8.
    let mut rng = SplitMix64::new(0x0706_0301);
    let worst = [
        Worst::new("f32 M2L tables vs f64, p ≤ 8 (terms)"),
        Worst::new("f32 M2M tables vs f64, p ≤ 8 (terms)"),
        Worst::new("f32 L2L tables vs f64, p ≤ 8 (terms)"),
    ];
    for p in 0..=P_MAX_F32 {
        let single = RotationTables::<f32>::build(p);
        let mut scratch = RotationScratch::<f32>::new(p);
        for (op, worst) in OPERATORS.into_iter().zip(&worst) {
            for t in 0..op.count() {
                for _ in 0..2 {
                    let input: Vec<f32> = random_coefficients(input_kind(op), p, &mut rng)
                        .iter()
                        .map(|&v| v as f32)
                        .collect();
                    let input64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
                    let mut got = vec![0.0f32; len(p)];
                    single.tables(op).apply(t, &input, &mut got, &mut scratch);
                    let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
                    let want = run(tables(p).tables(op), t, &input64);
                    let e = worst.update(error(op, dense(p).of(op), t, &input64, &got, &want));
                    assert!(e <= F32_TOL, "{op:?}, p = {p}, entry {t}: {e:.3e}");
                }
            }
        }
    }
}

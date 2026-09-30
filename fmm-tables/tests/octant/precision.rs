//! f32 tables: the f64 table rounded entry by entry, applied in f32, against the f64
//! table applied in f64.

use nd_fmm_tables::geometry::OCTANT_COUNT;
use nd_fmm_tables::{L2lTables, M2mTables};

use crate::common::{
    Kind, P_MAX_F32, SplitMix64, Worst, apply, degree_error, l2l, len, m2m, random_coefficients,
    terms,
};

/// Tolerance of the f32 tables, per degree relative to the terms (brief T3).
const F32_TOL: f64 = 1e-5;

#[test]
fn f32_tables_match_f64_to_single_precision() {
    // The input is rounded to f32 first, so both runs see the same coefficients.
    //
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for M2M,
    // Nₘ/Sₘ for L2L), relative to the terms |Aᵢₖ xₖ| of the f64 table application.
    let mut rng = SplitMix64::new(0x0c7a_0201);
    let worst_m2m = Worst::new("f32 M2M table vs f64, per degree (terms)");
    let worst_l2l = Worst::new("f32 L2L table vs f64, per degree (terms)");
    for p in 0..=P_MAX_F32 {
        let single = (M2mTables::<f32>::build(p), L2lTables::<f32>::build(p));
        for o in 0..OCTANT_COUNT {
            for (kind, worst, double, apply_single) in [
                (
                    Kind::Multipole,
                    &worst_m2m,
                    m2m(p).matrices(),
                    &(|x: &[f32], y: &mut [f32]| single.0.apply(o, x, y))
                        as &dyn Fn(&[f32], &mut [f32]),
                ),
                (
                    Kind::Local,
                    &worst_l2l,
                    l2l(p).matrices(),
                    &(|x: &[f32], y: &mut [f32]| single.1.apply(o, x, y))
                        as &dyn Fn(&[f32], &mut [f32]),
                ),
            ] {
                for _ in 0..4 {
                    let input: Vec<f32> = random_coefficients(kind, p, &mut rng)
                        .iter()
                        .map(|&v| v as f32)
                        .collect();
                    let input64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
                    let mut got = vec![0.0f32; len(p)];
                    apply_single(&input, &mut got);
                    let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
                    let want = apply(double, o, &input64);
                    let scale = terms(double, o, &input64);
                    let e = worst.update(degree_error(kind, p, &got, &want, &scale));
                    assert!(e <= F32_TOL, "{kind:?}, o = {o}, p = {p}: {e:.3e}");
                }
            }
        }
    }
}

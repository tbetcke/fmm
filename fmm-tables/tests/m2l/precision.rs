//! f32 tables: the f64 table rounded entry by entry, applied in f32, against the f64
//! table applied in f64.

use nd_fmm_tables::M2lTables;
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;

use crate::common::{
    DEBUG_DEGREES, Kind, P_MAX_F32, SplitMix64, Worst, apply, degree_error, len, m2l,
    random_coefficients, terms,
};

/// Tolerance of the f32 tables, per degree relative to the terms (brief T4).
const F32_TOL: f64 = 1e-5;

#[test]
fn f32_tables_are_the_f64_tables_rounded_and_match_f64_to_single_precision() {
    // Entries: exact equality of every f32 entry with `as f32` of the f64 entry.
    //
    // Application: the input is rounded to f32 first, so both runs see the same
    // coefficients. Error measure: local coefficients per degree in the orthonormal
    // weighting Nₘ/Sₘ, relative to the terms |Aᵢₖ xₖ| of the f64 table application.
    // All 316 offsets, two inputs each, p ≤ 8.
    let mut rng = SplitMix64::new(0x4d21_0201);
    let worst = Worst::new("f32 M2L table vs f64, per degree (terms)");
    for p in DEBUG_DEGREES.into_iter().filter(|&p| p <= P_MAX_F32) {
        let single = M2lTables::<f32>::build(p);
        let double = m2l(p).matrices();
        assert_eq!(single.p(), p);
        for (s, d) in single.matrices().as_slice().iter().zip(double.as_slice()) {
            assert_eq!(s.to_bits(), (*d as f32).to_bits());
        }
        for index in 0..M2L_OFFSET_COUNT {
            for _ in 0..2 {
                let input: Vec<f32> = random_coefficients(Kind::Multipole, p, &mut rng)
                    .iter()
                    .map(|&v| v as f32)
                    .collect();
                let input64: Vec<f64> = input.iter().map(|&v| f64::from(v)).collect();
                let mut got = vec![0.0f32; len(p)];
                single.apply(index, &input, &mut got);
                let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
                let want = apply(double, index, &input64);
                let scale = terms(double, index, &input64);
                let e = worst.update(degree_error(Kind::Local, p, &got, &want, &scale));
                assert!(e <= F32_TOL, "index {index}, p = {p}: {e:.3e}");
            }
        }
    }
}

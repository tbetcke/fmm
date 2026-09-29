//! Real storage of P2M and P2L (CONVENTIONS §3.6, §3.7): one unit charge gives the
//! conjugated harmonic of its scaled position.

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Workspace, leaf};

use crate::common::{P_MAX, SplitMix64, frames, len, place};

/// conj(X) in real storage: X with the slots m < 0 (imaginary parts) negated.
fn conjugate(p: usize, mut values: Vec<f64>) -> Vec<f64> {
    let layout = Layout::new(p);
    for (i, v) in values.iter_mut().enumerate() {
        if layout.nm(i).1 < 0 {
            *v = -*v;
        }
    }
    values
}

fn assert_bits_equal(got: &[f64], want: &[f64], what: &str) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}, slot {i}: {g} vs {w}");
    }
}

#[test]
fn p2m_of_unit_charge_is_conjugate_regular_harmonic() {
    // Error measure: bit-for-bit equality with R(u), u = frame.scaled(y), with the −m
    // slots negated. The points are generic (no zero coordinate), so no signed zero
    // arises. Sources cover |u| ≤ √3 plus a margin (CONVENTIONS §3.9).
    let mut rng = SplitMix64::new(0x7031);
    let mut ws = Workspace::new(P_MAX);
    for frame in frames() {
        for _ in 0..8 {
            let y = place(&frame, rng.in_shell(0.05, 1.8));
            for p in 0..=P_MAX {
                let mut multipole = vec![0.0; len(p)];
                leaf::p2m(p, &frame, &[y], &[1.0], &mut ws, &mut multipole);
                let mut regular = vec![0.0; len(p)];
                harmonics::regular(p, frame.scaled(y), &mut regular);
                assert_bits_equal(&multipole, &conjugate(p, regular), "P2M");
            }
        }
    }
}

#[test]
fn p2l_of_unit_charge_is_conjugate_irregular_harmonic() {
    // Error measure: bit-for-bit equality with I(u), u = frame.scaled(y), with the −m
    // slots negated. Sources cover |u| ≥ 2 minus a margin (CONVENTIONS §3.9).
    let mut rng = SplitMix64::new(0x7032);
    let mut ws = Workspace::new(P_MAX);
    for frame in frames() {
        for _ in 0..8 {
            let y = place(&frame, rng.in_shell(1.9, 8.0));
            for p in 0..=P_MAX {
                let mut local = vec![0.0; len(p)];
                leaf::p2l(p, &frame, &[y], &[1.0], &mut ws, &mut local);
                let mut irregular = vec![0.0; len(p)];
                harmonics::irregular(p, frame.scaled(y), &mut irregular);
                assert_bits_equal(&local, &conjugate(p, irregular), "P2L");
            }
        }
    }
}

#[test]
fn unit_charge_at_low_degree_matches_conventions_examples() {
    // CONVENTIONS §3.3: R₁⁰ = z, R₁¹ = (x + iy)/2, I₀⁰ = 1/r, I₁⁰ = z/r³, I₁¹ = (x + iy)/r³.
    // With conjugation, slot −1 holds −Im. The frame and point are dyadic, so
    // u = (1, 2, 2) and |u| = 3 exactly. Error measure: exact equality for P2M, whose
    // values are dyadic; absolute error 4 ε |value| for P2L, whose values involve
    // thirds and are rounded a few times.
    let frame = nd_fmm_ref::Frame::new([1.0, -1.0, 0.5], 0.25);
    let y = [1.25, -0.5, 1.0];
    assert_eq!(frame.scaled(y), [1.0, 2.0, 2.0]);
    let mut ws = Workspace::new(1);
    let mut multipole = [0.0; 4];
    leaf::p2m(1, &frame, &[y], &[1.0], &mut ws, &mut multipole);
    // Slots in storage order: (0, 0), (1, −1), (1, 0), (1, 1).
    assert_eq!(multipole, [1.0, -1.0, 2.0, 0.5]);
    let mut local = [0.0; 4];
    leaf::p2l(1, &frame, &[y], &[1.0], &mut ws, &mut local);
    let want: [f64; 4] = [1.0 / 3.0, -2.0 / 27.0, 2.0 / 27.0, 1.0 / 27.0];
    for (got, want) in local.iter().zip(want) {
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
            "{got} vs {want}"
        );
    }
}

//! The tables themselves: canonical columns against `direct`, the block structure of
//! M2M and L2L, hand-worked degrees 0 and 1, the leading block and determinism.

use nd_fmm_math::Layout;
use nd_fmm_ref::{Workspace, direct};
use nd_fmm_tables::geometry::{OCTANT_COUNT, l2l_frames, m2m_frames, octant_direction};
use nd_fmm_tables::{L2lTables, M2mTables, MatrixSet};

use crate::common::{P_DEBUG, l2l, len, m2m};

/// Bit patterns of every entry.
fn bits(set: &MatrixSet<f64>) -> Vec<u64> {
    set.as_slice().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn canonical_columns_equal_direct_bit_for_bit() {
    // Error measure: bit-for-bit equality of column k of matrix o with the `direct`
    // operator applied to the k-th unit vector at the canonical frames of §3.12.
    for p in [0, 1, 2, 5, 8] {
        let mut ws = Workspace::new(p);
        let (m2m, l2l) = (m2m(p), l2l(p));
        assert_eq!((m2m.p(), l2l.p()), (p, p));
        for (set, name) in [(m2m.matrices(), "M2M"), (l2l.matrices(), "L2L")] {
            assert_eq!((set.n(), set.count()), (len(p), OCTANT_COUNT), "{name}");
            assert_eq!(set.as_slice().len(), 8 * len(p) * len(p), "{name}");
        }
        for o in 0..OCTANT_COUNT {
            for k in 0..len(p) {
                let mut unit = vec![0.0; len(p)];
                unit[k] = 1.0;
                let column = |set: &MatrixSet<f64>| -> Vec<u64> {
                    set.matrix(o)[k * len(p)..(k + 1) * len(p)]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect()
                };
                let (from, to) = m2m_frames(o);
                let mut want = vec![0.0; len(p)];
                direct::m2m(p, &from, &to, &mut ws, &unit, &mut want);
                let want: Vec<u64> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    column(m2m.matrices()),
                    want,
                    "M2M p = {p}, o = {o}, k = {k}"
                );

                let (from, to) = l2l_frames(o);
                let mut want = vec![0.0; len(p)];
                direct::l2l(p, &from, &to, &mut ws, &unit, &mut want);
                let want: Vec<u64> = want.iter().map(|v| v.to_bits()).collect();
                assert_eq!(
                    column(l2l.matrices()),
                    want,
                    "L2L p = {p}, o = {o}, k = {k}"
                );
            }
        }
    }
}

#[test]
fn m2m_is_block_lower_and_l2l_block_upper_triangular() {
    // CONVENTIONS §3.11: M2M output degree j uses input degrees k ≤ j only, and L2L
    // output degree j uses input degrees n ≥ j only.
    //
    // Error measure: exact equality with zero outside those blocks.
    for p in 0..=P_DEBUG {
        let layout = Layout::new(p);
        let n = layout.len();
        for o in 0..OCTANT_COUNT {
            let (a, b) = (m2m(p).matrices().matrix(o), l2l(p).matrices().matrix(o));
            for col in 0..n {
                for row in 0..n {
                    let (j, k) = (layout.nm(row).0, layout.nm(col).0);
                    if k > j {
                        assert_eq!(
                            a[row + col * n],
                            0.0,
                            "M2M p = {p}, o = {o}, ({row}, {col})"
                        );
                    }
                    if k < j {
                        assert_eq!(
                            b[row + col * n],
                            0.0,
                            "L2L p = {p}, o = {o}, ({row}, {col})"
                        );
                    }
                }
                // The diagonal blocks are ρʲ I and σʲ⁺¹ I with ρ = σ = ½ (zero shift
                // within one degree): the only same-degree term is k = j, l = i.
                let (j, _) = layout.nm(col);
                for row in layout.degree(j) {
                    let delta = if row == col { 1.0 } else { 0.0 };
                    assert_eq!(a[row + col * n], delta * 0.5f64.powi(j as i32));
                    assert_eq!(b[row + col * n], delta * 0.5f64.powi(j as i32 + 1));
                }
            }
        }
    }
}

#[test]
fn degrees_zero_and_one_match_hand_computed_values() {
    // p = 0: M2M gives M̃′₀⁰ = M̃₀⁰, so the table is [1]; L2L gives
    // L̃′₀⁰ = σ L̃₀⁰ = ½ L̃₀⁰, so the table is [½].
    //
    // p = 1, slots 0: (0, 0), 1: Im order 1, 2: (1, 0), 3: Re order 1 (§3.6), with
    // R₁⁰(x) = z and R₁¹(x) = (x + iy)/2 (§3.3), and s = s_o.
    //
    // M2M (§3.11) with b = ½ s and ρ = ½: M̃′₀⁰ = M̃₀⁰; M̃′₁⁰ = b_z M̃₀⁰ + ½ M̃₁⁰; and
    // M̃′₁¹ = conj R₁¹(b) M̃₀⁰ + ½ M̃₁¹ with conj R₁¹(b) = (b_x − i b_y)/2. In real
    // storage, with b = ½ s,
    //
    //   [ 1        0  0  0 ]
    //   [ −s_y/4   ½  0  0 ]
    //   [  s_z/2   0  ½  0 ]
    //   [  s_x/4   0  0  ½ ].
    //
    // L2L (§3.11) with t = ½ s and σ = ½: L̃′₀⁰ = σ (L̃₀⁰ + Σₘ L̃₁ᵐ R₁ᵐ(t)), where by the
    // doubling rule of §3.6 Σₘ L̃₁ᵐ R₁ᵐ(t) = t_z L̃₁⁰ + 2 (Re L̃₁¹ t_x/2 − Im L̃₁¹ t_y/2);
    // and L̃′₁ⁱ = σ² L̃₁ⁱ. In real storage, with t = ½ s,
    //
    //   [ ½  −s_y/4  s_z/4  s_x/4 ]
    //   [ 0   ¼      0      0     ]
    //   [ 0   0      ¼      0     ]
    //   [ 0   0      0      ¼     ].
    //
    // Error measure: exact equality; every entry is a dyadic rational, and each is
    // formed by at most one rounding-free product.
    assert_eq!(M2mTables::<f64>::build(0).matrices().as_slice(), [1.0; 8]);
    assert_eq!(L2lTables::<f64>::build(0).matrices().as_slice(), [0.5; 8]);
    for o in 0..OCTANT_COUNT {
        let [sx, sy, sz] = octant_direction(o).map(|s| s as f64);
        // Row-major, as written above; the tables are column-major.
        let m2m_rows = [
            [1.0, 0.0, 0.0, 0.0],
            [-sy / 4.0, 0.5, 0.0, 0.0],
            [sz / 2.0, 0.0, 0.5, 0.0],
            [sx / 4.0, 0.0, 0.0, 0.5],
        ];
        let l2l_rows = [
            [0.5, -sy / 4.0, sz / 4.0, sx / 4.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ];
        for (set, rows, name) in [
            (m2m(1).matrices(), m2m_rows, "M2M"),
            (l2l(1).matrices(), l2l_rows, "L2L"),
        ] {
            for (r, row) in rows.iter().enumerate() {
                for (c, want) in row.iter().enumerate() {
                    assert_eq!(set.matrix(o)[r + 4 * c], *want, "{name} o = {o} ({r}, {c})");
                }
            }
        }
    }
}

#[test]
fn table_at_p_is_the_leading_block_of_the_table_at_p_plus_3() {
    // CONVENTIONS §3.12: the entries do not depend on p.
    //
    // Error measure: per entry, |a − b| ≤ 4 ε |b| (bit-for-bit equality passes); the
    // test reports whether every entry is bit-identical.
    let mut identical = true;
    let mut worst: f64 = 0.0;
    for p in 0..=P_DEBUG - 3 {
        let (n, big) = (len(p), len(p + 3));
        for (small, large, name) in [
            (m2m(p).matrices(), m2m(p + 3).matrices(), "M2M"),
            (l2l(p).matrices(), l2l(p + 3).matrices(), "L2L"),
        ] {
            for o in 0..OCTANT_COUNT {
                for c in 0..n {
                    for r in 0..n {
                        let a = small.matrix(o)[r + c * n];
                        let b = large.matrix(o)[r + c * big];
                        identical &= a.to_bits() == b.to_bits();
                        let e = (a - b).abs();
                        if b != 0.0 {
                            worst = worst.max(e / b.abs());
                        }
                        assert!(
                            e <= 4.0 * f64::EPSILON * b.abs(),
                            "{name} p = {p}, o = {o}, ({r}, {c}): {a} vs {b}"
                        );
                    }
                }
            }
        }
    }
    eprintln!("leading block: bit-identical = {identical}, worst relative difference {worst:.3e}");
}

#[test]
fn building_twice_is_bit_identical() {
    // Error measure: bit-for-bit equality of every entry, in f64 and f32.
    for p in [0, 3, 7] {
        assert_eq!(
            bits(M2mTables::build(p).matrices()),
            bits(m2m(p).matrices())
        );
        assert_eq!(
            bits(L2lTables::build(p).matrices()),
            bits(l2l(p).matrices())
        );
        let (a, b) = (M2mTables::<f32>::build(p), M2mTables::<f32>::build(p));
        assert!(
            a.matrices()
                .as_slice()
                .iter()
                .zip(b.matrices().as_slice())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
        let (a, b) = (L2lTables::<f32>::build(p), L2lTables::<f32>::build(p));
        assert!(
            a.matrices()
                .as_slice()
                .iter()
                .zip(b.matrices().as_slice())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }
}

#[test]
fn f32_tables_are_the_f64_tables_rounded() {
    // Error measure: exact equality of every entry with `as f32` of the f64 entry.
    for p in [1, 4, 8] {
        let single = M2mTables::<f32>::build(p);
        for (s, d) in single
            .matrices()
            .as_slice()
            .iter()
            .zip(m2m(p).matrices().as_slice())
        {
            assert_eq!(s.to_bits(), (*d as f32).to_bits());
        }
        let single = L2lTables::<f32>::build(p);
        for (s, d) in single
            .matrices()
            .as_slice()
            .iter()
            .zip(l2l(p).matrices().as_slice())
        {
            assert_eq!(s.to_bits(), (*d as f32).to_bits());
        }
        assert_eq!(single.p(), p);
    }
}

#[test]
fn apply_accumulates_and_checks_lengths() {
    // Error measure: bit-for-bit equality with `MatrixSet::apply` from the same start.
    let p = 3;
    let tables = m2m(p);
    let x: Vec<f64> = (0..len(p)).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let y0: Vec<f64> = (0..len(p)).map(|i| i as f64 - 4.0).collect();
    for o in 0..OCTANT_COUNT {
        let (mut got, mut want) = (y0.clone(), y0.clone());
        tables.apply(o, &x, &mut got);
        tables.matrices().apply(o, &x, &mut want);
        assert_eq!(got, want);
        let (mut got, mut want) = (y0.clone(), y0.clone());
        l2l(p).apply(o, &x, &mut got);
        l2l(p).matrices().apply(o, &x, &mut want);
        assert_eq!(got, want);
    }
}

#[test]
#[should_panic(expected = "`x` must have length n = 16")]
fn apply_rejects_wrong_input_length() {
    m2m(3).apply(0, &[0.0; 9], &mut [0.0; 16]);
}

#[test]
#[should_panic(expected = "matrix index 8 out of range for 8 matrices")]
fn apply_rejects_octant_8() {
    l2l(3).apply(8, &[0.0; 16], &mut [0.0; 16]);
}

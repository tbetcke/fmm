//! The tables themselves: canonical columns against `direct`, partial builds, the
//! monopole column against P2L, the leading block, inversion, determinism and the
//! application interface.

use nd_fmm_math::Layout;
use nd_fmm_ref::{Workspace, direct, leaf};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_frames, m2l_offset_index, m2l_offsets};
use nd_fmm_tables::m2l::build_matrices;
use nd_fmm_tables::{M2lTables, MatrixSet};

use crate::common::{
    CHAIN_TOL, DEBUG_DEGREES, Kind, P_SUBSET, Worst, degree, degree_error, len, m2l, subset, terms,
};

/// Bit patterns of every entry.
fn bits(set: &MatrixSet<f64>) -> Vec<u64> {
    set.as_slice().iter().map(|v| v.to_bits()).collect()
}

/// Checks columns `columns` of matrix `position` of `set`, the table of index `index`,
/// against `direct::m2l` of the unit vectors at the canonical frames, bit for bit.
fn check_columns(
    set: &MatrixSet<f64>,
    position: usize,
    index: usize,
    columns: impl Iterator<Item = usize>,
    ws: &mut Workspace<f64>,
) {
    let p = degree(set);
    let n = len(p);
    let (source, target) = m2l_frames(index);
    for k in columns {
        let mut unit = vec![0.0; n];
        unit[k] = 1.0;
        let mut want = vec![0.0; n];
        direct::m2l(p, &source, &target, ws, &unit, &mut want);
        let want: Vec<u64> = want.iter().map(|v| v.to_bits()).collect();
        let got: Vec<u64> = set.matrix(position)[k * n..(k + 1) * n]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want, "p = {p}, index {index}, column {k}");
    }
}

#[test]
fn canonical_columns_equal_direct_bit_for_bit() {
    // Error measure: bit-for-bit equality of column k of matrix index(d) with
    // `direct::m2l` of the k-th unit vector at the canonical frames of §3.12, source
    // (0, 1) and target (2d, 1). Every column of every offset at p ≤ 5; every eighth
    // offset at p = 8; every seventh column of the p = 12 subset (debug-run time).
    for p in [0, 1, 2, 3, 5, 8] {
        let mut ws = Workspace::new(p);
        let tables = m2l(p);
        assert_eq!(tables.p(), p);
        let set = tables.matrices();
        assert_eq!((set.n(), set.count()), (len(p), M2L_OFFSET_COUNT));
        assert_eq!(set.as_slice().len(), 316 * len(p) * len(p));
        let step = if p == 8 { 8 } else { 1 };
        for index in (0..M2L_OFFSET_COUNT).step_by(step) {
            check_columns(set, index, index, 0..len(p), &mut ws);
        }
    }
    let subset = subset();
    let mut ws = Workspace::new(P_SUBSET);
    for (position, &index) in subset.indices.iter().enumerate() {
        check_columns(
            &subset.matrices,
            position,
            index,
            (0..len(P_SUBSET)).step_by(7),
            &mut ws,
        );
    }
}

#[test]
fn partial_builds_equal_the_full_table_bit_for_bit() {
    // `build_matrices` gives matrix `indices[t]` of the full table as matrix t, in any
    // order and with repeats.
    //
    // Error measure: bit-for-bit equality of every entry.
    let indices = [315, 0, 17, 158, 17, 157];
    for p in [0, 3, 5] {
        let full = m2l(p).matrices();
        let part = build_matrices::<f64>(p, &indices);
        assert_eq!((part.n(), part.count()), (len(p), indices.len()));
        for (t, &index) in indices.iter().enumerate() {
            let got: Vec<u64> = part.matrix(t).iter().map(|v| v.to_bits()).collect();
            let want: Vec<u64> = full.matrix(index).iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "p = {p}, index {index}");
        }
        let single = build_matrices::<f32>(p, &indices);
        for (s, d) in single.as_slice().iter().zip(part.as_slice()) {
            assert_eq!(s.to_bits(), (*d as f32).to_bits());
        }
    }
    assert_eq!(build_matrices::<f64>(2, &[]).count(), 0);
}

#[test]
#[should_panic(expected = "M2L offset index 316 out of range for 316 offsets")]
fn partial_build_rejects_index_316() {
    let _ = build_matrices::<f64>(1, &[3, M2L_OFFSET_COUNT]);
}

/// Column 0 of matrix `position` of `set`, the table of index `index`, against P2L of a
/// unit charge at the source centre in the target frame; returns the per-degree error
/// relative to the terms |Aᵢ₀|.
fn monopole_error(
    set: &MatrixSet<f64>,
    position: usize,
    index: usize,
    ws: &mut Workspace<f64>,
) -> f64 {
    let p = degree(set);
    let n = len(p);
    let (source, target) = m2l_frames(index);
    let column = &set.matrix(position)[..n];
    let mut want = vec![0.0; n];
    leaf::p2l(p, &target, &[source.centre], &[1.0], ws, &mut want);
    let mut unit = vec![0.0; n];
    unit[0] = 1.0;
    let scale = terms(set, position, &unit);
    degree_error(Kind::Local, p, column, &want, &scale)
}

#[test]
fn monopole_column_equals_p2l_of_a_unit_charge() {
    // CONVENTIONS §3.11, "M2L", Truncation: for a charge at the source centre the
    // input is a unit monopole, and the M2L sum is the P2L coefficient
    // σʲ⁺¹ conj Iⱼⁱ(u − b) = conj Iⱼⁱ(−2d) in the target frame, with no truncation.
    //
    // Error measure: local coefficients per degree in the orthonormal weighting
    // Nₘ/Sₘ, relative to the terms |Aᵢ₀| of column 0. All 316 offsets at p ≤ 8, and
    // the p = 12 subset.
    let worst = Worst::new("M2L monopole column vs P2L, per degree (terms)");
    for p in DEBUG_DEGREES {
        let mut ws = Workspace::new(p);
        for index in 0..M2L_OFFSET_COUNT {
            let e = worst.update(monopole_error(m2l(p).matrices(), index, index, &mut ws));
            assert!(e <= CHAIN_TOL, "p = {p}, index {index}: {e:.3e}");
        }
    }
    let subset = subset();
    let mut ws = Workspace::new(P_SUBSET);
    for (position, &index) in subset.indices.iter().enumerate() {
        let e = worst.update(monopole_error(&subset.matrices, position, index, &mut ws));
        assert!(e <= CHAIN_TOL, "p = {P_SUBSET}, index {index}: {e:.3e}");
    }
}

/// Compares every matrix of `small` with the leading block of the matrix at the same
/// position of `large`; returns whether all entries are bit-identical, and the worst
/// relative difference. Asserts |a − b| ≤ 4 ε |b| per entry.
fn check_leading_block(small: &MatrixSet<f64>, large: &MatrixSet<f64>) -> (bool, f64) {
    let (n, big) = (small.n(), large.n());
    let mut identical = true;
    let mut worst: f64 = 0.0;
    for t in 0..small.count() {
        for c in 0..n {
            for r in 0..n {
                let a = small.matrix(t)[r + c * n];
                let b = large.matrix(t)[r + c * big];
                identical &= a.to_bits() == b.to_bits();
                let e = (a - b).abs();
                if b != 0.0 {
                    worst = worst.max(e / b.abs());
                }
                assert!(
                    e <= 4.0 * f64::EPSILON * b.abs(),
                    "p = {}, matrix {t}, ({r}, {c}): {a} vs {b}",
                    degree(small)
                );
            }
        }
    }
    (identical, worst)
}

#[test]
fn table_at_p_is_the_leading_block_of_the_table_at_p_plus_5() {
    // CONVENTIONS §3.12: the entries do not depend on p. Pairs (0, 5) and (3, 8) of
    // full tables, and (7, 12) on the offsets of the p = 12 subset.
    //
    // Error measure: per entry, |a − b| ≤ 4 ε |b| (bit-for-bit equality passes); the
    // test reports whether every entry is bit-identical.
    let mut identical = true;
    let mut worst: f64 = 0.0;
    for p in [0, 3] {
        let (i, w) = check_leading_block(m2l(p).matrices(), m2l(p + 5).matrices());
        identical &= i;
        worst = worst.max(w);
    }
    let subset = subset();
    let small = build_matrices::<f64>(P_SUBSET - 5, &subset.indices);
    let (i, w) = check_leading_block(&small, &subset.matrices);
    identical &= i;
    worst = worst.max(w);
    eprintln!("leading block: bit-identical = {identical}, worst relative difference {worst:.3e}");
}

/// The outcome of an entry-by-entry comparison that allows rounding: how many entries
/// differ in their bits, how many of those are zeros of opposite sign, and the worst
/// relative difference of the nonzero ones.
#[derive(Default)]
pub struct BitReport {
    pub entries: usize,
    pub differing: usize,
    pub signed_zeros: usize,
    pub worst: f64,
}

impl BitReport {
    /// Records the entry `got` against `want`.
    fn record(&mut self, got: f64, want: f64) {
        self.entries += 1;
        if got.to_bits() != want.to_bits() {
            self.differing += 1;
            if got == 0.0 && want == 0.0 {
                self.signed_zeros += 1;
            }
        }
        if want != 0.0 {
            self.worst = self.worst.max((got - want).abs() / want.abs());
        }
    }

    fn merge(&mut self, other: Self) {
        self.entries += other.entries;
        self.differing += other.differing;
        self.signed_zeros += other.signed_zeros;
        self.worst = self.worst.max(other.worst);
    }

    /// Prints the report under `name`.
    pub fn print(&self, name: &str) {
        eprintln!(
            "{name}: {} of {} entries differ in their bits, {} of them zeros of opposite \
             sign; worst relative difference {:.3e}",
            self.differing, self.entries, self.signed_zeros, self.worst
        );
    }
}

/// Checks the inversion identity M2L(−d) = diag((−1)ʲ) M2L(d) diag((−1)ⁿ) of
/// CONVENTIONS §3.12 for the 158 pairs ±d of the full table `set`, with j the output
/// and n the input degree. Asserts |a − b| ≤ 4 ε |b| per entry, and reports how far
/// the identity holds bit for bit.
pub fn check_inversion(set: &MatrixSet<f64>) -> BitReport {
    let p = degree(set);
    let layout = Layout::new(p);
    let n = layout.len();
    let offsets = m2l_offsets();
    let mut report = BitReport::default();
    let mut pairs = 0;
    for (index, d) in offsets.into_iter().enumerate() {
        let opposite = m2l_offset_index(d.map(|t| -t)).unwrap();
        // The lexicographic order is symmetric under d ↦ −d.
        assert_eq!(opposite, M2L_OFFSET_COUNT - 1 - index, "d = {d:?}");
        if opposite < index {
            continue;
        }
        pairs += 1;
        for c in 0..n {
            for r in 0..n {
                let sign = if (layout.nm(r).0 + layout.nm(c).0).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                };
                let a = set.matrix(opposite)[r + c * n];
                let b = sign * set.matrix(index)[r + c * n];
                report.record(a, b);
                assert!(
                    (a - b).abs() <= 4.0 * f64::EPSILON * b.abs(),
                    "p = {p}, d = {d:?}, ({r}, {c}): {a} vs {b}"
                );
            }
        }
    }
    assert_eq!(pairs, 158);
    report
}

#[test]
fn table_of_minus_d_is_the_parity_transform_of_the_table_of_d() {
    // CONVENTIONS §3.12, "Operator identities", P = −I:
    // M2L(−d) = diag((−1)ʲ) M2L(d) diag((−1)ⁿ), all 158 pairs ±d, p ≤ 8.
    //
    // Error measure: per entry, |a − b| ≤ 4 ε |b| (bit-for-bit equality passes); the
    // test reports how many entries differ in their bits, and how many of those are
    // zeros of opposite sign.
    let mut report = BitReport::default();
    for p in DEBUG_DEGREES {
        report.merge(check_inversion(m2l(p).matrices()));
    }
    report.print("inversion p ≤ 8");
}

#[test]
fn building_twice_is_bit_identical() {
    // Error measure: bit-for-bit equality of every entry, in f64 and f32.
    for p in [0, 3, 5] {
        assert_eq!(
            bits(M2lTables::build(p).matrices()),
            bits(m2l(p).matrices())
        );
    }
    let (a, b) = (M2lTables::<f32>::build(3), M2lTables::<f32>::build(3));
    assert!(
        a.matrices()
            .as_slice()
            .iter()
            .zip(b.matrices().as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits())
    );
    let indices = &subset().indices[..3];
    assert_eq!(
        bits(&build_matrices(P_SUBSET, indices)),
        bits(&build_matrices(P_SUBSET, indices))
    );
}

#[test]
fn index_delegates_to_the_geometry() {
    // Error measure: exact equality (integers).
    let tables = m2l(0);
    for (index, d) in m2l_offsets().into_iter().enumerate() {
        assert_eq!(tables.index(d), Some(index));
    }
    for d in [[0, 0, 0], [1, -1, 1], [4, 0, 0], [0, -4, 3]] {
        assert_eq!(tables.index(d), None, "d = {d:?}");
    }
}

#[test]
fn apply_accumulates_and_checks_lengths() {
    // Error measure: bit-for-bit equality with `MatrixSet::apply` from the same start.
    let p = 3;
    let tables = m2l(p);
    let x: Vec<f64> = (0..len(p)).map(|i| 1.0 / (1.0 + i as f64)).collect();
    let y0: Vec<f64> = (0..len(p)).map(|i| i as f64 - 4.0).collect();
    for index in 0..M2L_OFFSET_COUNT {
        let (mut got, mut want) = (y0.clone(), y0.clone());
        tables.apply(index, &x, &mut got);
        tables.matrices().apply(index, &x, &mut want);
        assert_eq!(got, want);
    }
}

#[test]
#[should_panic(expected = "`y` must have length n = 16")]
fn apply_rejects_wrong_output_length() {
    m2l(3).apply(0, &[0.0; 16], &mut [0.0; 9]);
}

#[test]
#[should_panic(expected = "matrix index 316 out of range for 316 matrices")]
fn apply_rejects_index_316() {
    m2l(3).apply(M2L_OFFSET_COUNT, &[0.0; 16], &mut [0.0; 16]);
}

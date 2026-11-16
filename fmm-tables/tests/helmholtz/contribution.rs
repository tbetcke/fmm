//! The M2L split into its contribution and its add (Helmholtz CH4.0i,
//! `HelmholtzTables::{m2l_contribution, add_m2l_contribution}`): the two in turn are `m2l`
//! bit for bit (`to_bits`, so a −0 differs from a +0), on every offset of two levels, one and
//! two multipole planes, one and two vectors, real and complex k, stored and per-call blocks
//! and raised M2L degrees; and several pairs' contributions added in a row's order are
//! successive `m2l` calls bit for bit, whatever order the contributions were computed in.

use nd_fmm_math::{Complex, Layout};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables, add_m2l_contribution};

use super::input;

fn assert_bits(got: &[f64], expected: &[f64], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: lengths");
    for (i, (a, b)) in got.iter().zip(expected).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{what}: value {i} is {a:e}, m2l gives {b:e}"
        );
    }
}

/// The tables of the tests: real and complex k, blocks stored and per call, and M2L degrees
/// raised on level 3.
fn tables() -> Vec<(String, HelmholtzTables<f64>)> {
    let mut out = Vec::new();
    for (re, im) in [(5.0, 0.0), (6.0, 0.5)] {
        let k = Complex::new(re, im);
        for budget in [DEFAULT_BLOCK_BUDGET, 0] {
            out.push((
                format!("k̂₀ = {re} + {im}i, budget {budget}"),
                HelmholtzTables::build(k, &[0, 0, 7, 6], budget),
            ));
        }
        out.push((
            format!("k̂₀ = {re} + {im}i, raised"),
            HelmholtzTables::build_with_m2l_degrees(
                k,
                &[0, 0, 7, 6],
                &[0, 0, 7, 9],
                DEFAULT_BLOCK_BUDGET,
            ),
        ));
    }
    out
}

#[test]
fn contribution_then_add_is_m2l() {
    for (name, tables) in tables() {
        let (mut a, mut b) = (tables.scratch(), tables.scratch());
        let planes: &[usize] = if tables.real_shifts() { &[1, 2] } else { &[2] };
        for level in 2..4 {
            let len = Layout::new(tables.m2l_degree(level)).len();
            for &planes in planes {
                for vectors in [1, 2] {
                    let size = vectors * 2 * len;
                    assert_eq!(tables.m2l_contribution_len(level, vectors), size);
                    for d in 0..M2L_OFFSET_COUNT {
                        let x = input(
                            tables.m2l_degree(level),
                            planes * vectors,
                            (d + 1000 * level) as u64,
                        );
                        // A local with values, a −0 among them.
                        let mut expected: Vec<f64> =
                            input(tables.m2l_degree(level), 2 * vectors, 7);
                        expected[0] = -0.0;
                        let mut got = expected.clone();
                        tables.m2l(level, d, &x, planes, &mut expected, vectors, &mut a);
                        // The slot holds garbage before: the contribution overwrites it.
                        let mut slot = vec![f64::NAN; size];
                        tables.m2l_contribution(level, d, &x, planes, &mut slot, vectors, &mut b);
                        add_m2l_contribution(&slot, &mut got);
                        assert_bits(
                            &got,
                            &expected,
                            &format!(
                                "{name}: level {level}, d = {d}, {planes} planes, m = {vectors}"
                            ),
                        );
                    }
                }
            }
        }
    }
}

/// A row of pairs: the contributions computed in reverse order into their slots, then added
/// in the row's order, against `m2l` pair after pair.
#[test]
fn a_row_added_in_order_is_successive_m2l_calls() {
    for (name, tables) in tables() {
        let mut scratch = tables.scratch();
        let level = 3;
        let q = tables.m2l_degree(level);
        let size = tables.m2l_contribution_len(level, 1);
        let row: Vec<(usize, Vec<f64>)> = (0..M2L_OFFSET_COUNT)
            .step_by(7)
            .map(|d| (d, input(q, 2, d as u64)))
            .collect();
        let mut expected = vec![0.0; size];
        for (d, x) in &row {
            tables.m2l(level, *d, x, 2, &mut expected, 1, &mut scratch);
        }
        let mut slots = vec![0.0; row.len() * size];
        for (i, (d, x)) in row.iter().enumerate().rev() {
            tables.m2l_contribution(
                level,
                *d,
                x,
                2,
                &mut slots[i * size..(i + 1) * size],
                1,
                &mut scratch,
            );
        }
        let mut got = vec![0.0; size];
        for slot in slots.chunks_exact(size) {
            add_m2l_contribution(slot, &mut got);
        }
        assert_bits(&got, &expected, &format!("{name}: the row"));
    }
}

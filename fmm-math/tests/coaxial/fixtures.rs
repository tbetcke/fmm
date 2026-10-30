//! Test 1: the tables against spike S2's arbitrary-precision fixture (120 sets: k a from
//! 0.01 to 300, p to 371, real, complex and imaginary k, M2M at the child shift and M2L
//! at d = 2 and √27 box widths).

use crate::support::{Op, Set, sets};

/// Bound on the FMM-weighted error against the fixture (S2's question 2, signed off
/// 2026-10-27).
const WEIGHTED: f64 = 1e-11;

/// Entries of the fixture below this are rounded to (near) zero in f64 and not compared
/// entry by entry.
const TINY: f64 = 1e-290;

/// The start column E⁰_{l0} = (−1)ˡ √(2l + 1) f̂ₗ, read from the recursion's table,
/// against the fixture's radial samples to max(p_src, p_tgt), relative 1e-13.
#[test]
fn start_column_agrees_with_the_fixture() {
    let mut worst = (0.0f64, String::new());
    for set in sets().iter().filter(|s| s.p_src.max(s.p_tgt) <= 100) {
        let table = set.recursion();
        for &(l, f) in &set.radial {
            if l > set.p_src.max(set.p_tgt) || f.norm() < TINY {
                continue;
            }
            let sign = if l % 2 == 1 { -1.0 } else { 1.0 };
            let exact = f * (sign * ((2 * l + 1) as f64).sqrt());
            let got = if l <= set.p_tgt {
                table.get(0, l, 0)
            } else {
                table.get(0, 0, l) * sign
            };
            let err = (got - exact).norm() / exact.norm();
            if err > worst.0 {
                worst = (err, format!("{} l={l}", set.name));
            }
        }
    }
    println!(
        "start column: worst relative error {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 <= 1e-13, "{worst:?}");
}

/// The rule's table and the recursion's at the set's sampled entries: (weighted error of
/// the rule's table, of the recursion's, per-entry relative error of the recursion's).
/// The weighted error is the largest |Δ| w over the samples divided by the largest |Ê| w
/// of the rule's full table.
fn errors(set: &Set) -> (f64, f64, f64) {
    let (input, output) = set.weights();
    let rule = set.by_rule();
    let recursion = set.recursion();
    let largest = |table: &nd_fmm_math::coaxial::CoaxialTable<f64>| {
        let mut out = 0.0f64;
        for m in 0..table.orders() {
            for (n1, &w_out) in output.iter().enumerate().take(set.p_tgt + 1).skip(m) {
                for (n, &w_in) in input.iter().enumerate().take(set.p_src + 1).skip(m) {
                    out = out.max(table.get(m, n1, n).norm() * w_out * w_in);
                }
            }
        }
        out
    };
    let den = largest(&rule);
    if set.recursion_stable() {
        // An independent table must give the same normalisation, so that a spurious
        // large entry of the rule's table cannot hide an error.
        let other = largest(&recursion);
        assert!(
            (den - other).abs() <= 1e-10 * den,
            "{}: weighted maxima {den:e} {other:e}",
            set.name
        );
    }
    let (mut rule_err, mut rec_err, mut entry) = (0.0f64, 0.0f64, 0.0f64);
    for &(m, n1, n, exact) in &set.entries {
        let (a, b) = (rule.get(m, n1, n), recursion.get(m, n1, n));
        assert!(
            a.re.is_finite() && a.im.is_finite(),
            "{} ({m}, {n1}, {n})",
            set.name
        );
        let w = output[n1] * input[n];
        rule_err = rule_err.max((a - exact).norm() * w);
        rec_err = rec_err.max((b - exact).norm() * w);
        if exact.norm() >= TINY {
            entry = entry.max((b - exact).norm() / exact.norm());
        }
    }
    (rule_err / den, rec_err / den, entry)
}

/// Checks every set with max(p_src, p_tgt) ≤ `p_max`: the rule's table within the
/// weighted bound everywhere, the recursion's where S2 found it stable, and the
/// recursion's entries within 1e-12 relative at k a ≤ 3 (signed off 2026-10-30: on CI's
/// glibc the e^{iz} of the start column differs from macOS's in the last bit, and an
/// entry came to 1.4e-13 against 8.1e-14 on the M3 Max).
fn check(p_max: usize) {
    let mut worst_rule = (0.0f64, String::new());
    let mut worst_rec = (0.0f64, String::new());
    let mut worst_entry = (0.0f64, String::new());
    for set in sets().iter().filter(|s| s.p_src.max(s.p_tgt) <= p_max) {
        let (rule, rec, entry) = errors(set);
        println!(
            "{:<38} {} p {:>3} -> {:>3}: rule {rule:.1e}, recursion {rec:.1e} (entries {entry:.1e})",
            set.name,
            if set.op == Op::Regular { "RR" } else { "SR" },
            set.p_src,
            set.p_tgt
        );
        if rule > worst_rule.0 {
            worst_rule = (rule, set.name.clone());
        }
        if set.recursion_stable() && rec > worst_rec.0 {
            worst_rec = (rec, set.name.clone());
        }
        if set.ka <= 3.0 && entry > worst_entry.0 {
            worst_entry = (entry, set.name.clone());
        }
    }
    println!(
        "worst: rule {:.1e} ({}), recursion where stable {:.1e} ({}), entries at k a ≤ 3 \
         {:.1e} ({})",
        worst_rule.0, worst_rule.1, worst_rec.0, worst_rec.1, worst_entry.0, worst_entry.1
    );
    assert!(worst_rule.0 <= WEIGHTED, "{worst_rule:?}");
    assert!(worst_rec.0 <= WEIGHTED, "{worst_rec:?}");
    assert!(worst_entry.0 <= 1e-12, "{worst_entry:?}");
}

/// The sets to p = 100 (k a ≤ 30), where the rule uses the recursion.
#[test]
fn tables_agree_with_the_fixture_to_degree_100() {
    check(100);
}

/// Every set, the projections of the rule at k a = 100 and 300 included.
#[test]
#[ignore = "release mode: tables to p = 371"]
fn tables_agree_with_the_fixture() {
    check(usize::MAX);
}

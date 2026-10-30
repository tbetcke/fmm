//! Test 1: every row of the mpmath fixture of tools/fixtures/gen_helmholtz_radial.py
//! (fmm-math/fixtures/helmholtz_radial.json, spike S3's sets), values and derivatives,
//! on every operator's range.

use std::collections::BTreeMap;

use nd_fmm_math::{CONVENTION_VERSION, Complex};
use serde_json::Value;

use crate::support::{Family, evaluate};

/// Bound on the error: relative for h, i, k; relative to the oscillation envelope for j.
const BOUND: f64 = 1e-13;

/// Reference values below the f64 range are stored as 0; ours must lie below this.
const UNDERFLOW: f64 = 1e-280;

fn fixture() -> Value {
    let text = include_str!("../../fixtures/helmholtz_radial.json");
    let doc: Value = serde_json::from_str(text).expect("the fixture is JSON");
    assert_eq!(
        doc["header"]["convention_version"].as_u64(),
        Some(CONVENTION_VERSION as u64),
        "regenerate the fixture: tools/fixtures/README.md"
    );
    doc
}

fn number(v: &Value) -> f64 {
    v.as_f64().expect("a number")
}

/// An exact double stored as a decimal string (z and s), parsed by `str::parse`, which
/// rounds correctly: one ulp of z moves e^{iz} by |z| ulp.
fn exact(v: &Value) -> f64 {
    v.as_str()
        .expect("a decimal string")
        .parse()
        .expect("an f64")
}

/// The error of one value: 0 if both underflow, else relative to `scale`; `None` if the
/// reference underflows and ours does not.
fn error(ours: Complex<f64>, reference: Complex<f64>, scale: f64) -> Option<f64> {
    if reference == Complex::zero() {
        return (ours.abs() < UNDERFLOW).then_some(0.0);
    }
    Some((ours - reference).abs() / scale)
}

#[test]
fn every_operator_range_agrees_with_the_fixture() {
    let doc = fixture();
    // (operator, family) → (worst value error, worst derivative error, rows, where)
    let mut worst: BTreeMap<(String, String), (f64, f64, usize, String)> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut rows_seen = 0;
    for set in doc["sets"].as_array().expect("sets") {
        let name = set["name"].as_str().expect("a name");
        let family_name = set["family"].as_str().expect("a family");
        let family = Family::from_name(family_name);
        let z = Complex::new(exact(&set["z"][0]), exact(&set["z"][1]));
        let s = exact(&set["s"]);
        let rows = set["rows"].as_array().expect("rows");
        let top = rows
            .iter()
            .map(|r| r[0].as_u64().expect("an order") as usize)
            .max()
            .expect("a row");
        let (v, d) = evaluate(family, z, s, top);
        let op = name.split('-').next().expect("an operator");
        let op = if op.starts_with("random") {
            "random"
        } else {
            op
        };
        let entry = worst
            .entry((op.to_string(), family_name.to_string()))
            .or_insert((0.0, 0.0, 0, String::new()));
        for r in rows {
            let n = r[0].as_u64().expect("an order") as usize;
            let rv = Complex::new(number(&r[1]), number(&r[2]));
            let rd = Complex::new(number(&r[3]), number(&r[4]));
            let (sv, sd) = if family == Family::J {
                (number(&r[5]), number(&r[6]))
            } else {
                (rv.abs(), rd.abs())
            };
            for (k, ours, reference, scale) in [(0, v[n], rv, sv), (1, d[n], rd, sd)] {
                match error(ours, reference, scale) {
                    Some(e) if e <= BOUND => {
                        let slot = if k == 0 { &mut entry.0 } else { &mut entry.1 };
                        if e > *slot {
                            *slot = e;
                            if k == 0 {
                                entry.3 = format!("{name} n={n}");
                            }
                        }
                    }
                    e => failures.push(format!(
                        "{name} n={n} {}: {ours:?} vs {reference:?} (error {e:?})",
                        if k == 0 { "value" } else { "derivative" }
                    )),
                }
            }
            entry.2 += 1;
            rows_seen += 1;
        }
    }
    println!("| operator | family | rows | worst value | worst derivative | at |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for ((op, family), (wv, wd, count, at)) in &worst {
        println!("| {op} | {family} | {count} | {wv:.1e} | {wd:.1e} | {at} |");
    }
    assert!(rows_seen > 13_000, "the fixture holds S3's 13,174 rows");
    assert!(
        failures.is_empty(),
        "{} values beyond {BOUND:e}:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
}

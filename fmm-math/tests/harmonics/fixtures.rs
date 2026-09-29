//! Values and gradients against the mpmath fixtures of tools/fixtures/
//! (fmm-math/fixtures/harmonics_{A,B}.json): set A in f64 to relative 1e-13 (p = 30),
//! set B in f32 to relative 1e-5 (p = 8).

use nd_fmm_math::harmonics::{irregular, irregular_grad, regular, regular_grad};
use nd_fmm_math::{CONVENTION_VERSION, Layout, RealScalar};
use serde_json::Value;

use crate::common::degree_relative_error;

struct Record {
    x: [f64; 3],
    value: Vec<f64>,
    grad: [Vec<f64>; 3],
}

struct Fixture {
    p: usize,
    regular: Vec<Record>,
    irregular: Vec<Record>,
}

fn number(v: &Value) -> f64 {
    v.as_str()
        .expect("fixture numbers are decimal strings")
        .parse()
        .expect("fixture numbers parse as f64")
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("expected an array")
        .iter()
        .map(number)
        .collect()
}

fn records(v: &Value, len: usize) -> Vec<Record> {
    v.as_array()
        .expect("expected a list of records")
        .iter()
        .map(|r| {
            let x = numbers(&r["x"]);
            let grad: Vec<Vec<f64>> = r["grad"]
                .as_array()
                .expect("grad is a list of three arrays")
                .iter()
                .map(numbers)
                .collect();
            let record = Record {
                x: [x[0], x[1], x[2]],
                value: numbers(&r["value"]),
                grad: grad.try_into().expect("grad has three components"),
            };
            assert_eq!(record.value.len(), len);
            assert!(record.grad.iter().all(|g| g.len() == len));
            record
        })
        .collect()
}

fn load(set: &str) -> Fixture {
    let path = format!(
        "{}/fixtures/harmonics_{set}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let json: Value = serde_json::from_str(&text).expect("fixture is valid JSON");
    let header = &json["header"];
    assert_eq!(header["set"], set);
    assert_eq!(
        header["convention_version"].as_u64(),
        Some(u64::from(CONVENTION_VERSION)),
        "fixture {set} was generated for another CONVENTION_VERSION"
    );
    let p = header["p"].as_u64().expect("header has p") as usize;
    let len = Layout::new(p).len();
    Fixture {
        p,
        regular: records(&json["regular"], len),
        irregular: records(&json["irregular"], len),
    }
}

type GradFn<T> = fn(usize, [T; 3], &mut [T], [&mut [T]; 3]);
type ValueFn<T> = fn(usize, [T; 3], &mut [T]);

/// Largest degree-relative error over the records of one family, over the value and
/// the three gradient components.
fn family_error<T: RealScalar>(
    p: usize,
    records: &[Record],
    value_fn: ValueFn<T>,
    grad_fn: GradFn<T>,
) -> f64 {
    let len = Layout::new(p).len();
    let mut value = vec![T::zero(); len];
    let mut only_value = vec![T::zero(); len];
    let [mut gx, mut gy, mut gz] = [0; 3].map(|_| vec![T::zero(); len]);
    let to_f64 = |v: &[T]| v.iter().map(|&c| c.to_f64()).collect::<Vec<_>>();
    let mut worst = 0.0_f64;
    for record in records {
        let x = record.x.map(T::from_f64);
        grad_fn(p, x, &mut value, [&mut gx, &mut gy, &mut gz]);
        value_fn(p, x, &mut only_value);
        assert!(
            value == only_value,
            "the gradient functions must return the same values"
        );
        worst = worst.max(degree_relative_error(p, &to_f64(&value), &record.value));
        for (g, reference) in [&gx, &gy, &gz].into_iter().zip(&record.grad) {
            worst = worst.max(degree_relative_error(p, &to_f64(g), reference));
        }
    }
    worst
}

fn check_set<T: RealScalar>(set: &str, expected_p: usize, tol: f64) {
    let fixture = load(set);
    assert_eq!(fixture.p, expected_p);
    let regular_err = family_error::<T>(fixture.p, &fixture.regular, regular, regular_grad);
    let irregular_err = family_error::<T>(fixture.p, &fixture.irregular, irregular, irregular_grad);
    println!("set {set}: regular {regular_err:.2e}, irregular {irregular_err:.2e}");
    assert!(
        regular_err <= tol,
        "set {set} regular: {regular_err:e} > {tol:e}"
    );
    assert!(
        irregular_err <= tol,
        "set {set} irregular: {irregular_err:e} > {tol:e}"
    );
}

#[test]
fn f64_matches_fixture_set_a() {
    check_set::<f64>("A", 30, 1e-13);
}

#[test]
fn f32_matches_fixture_set_b() {
    check_set::<f32>("B", 8, 1e-5);
}

/// Not required by the brief; set B has five times as many points as set A.
#[test]
fn f64_matches_fixture_set_b() {
    check_set::<f64>("B", 8, 1e-13);
}

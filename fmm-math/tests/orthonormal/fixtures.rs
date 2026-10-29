//! The harmonics, their derivatives and the rotation blocks against the mpmath fixture
//! of tools/fixtures/gen_helmholtz_angular.py (fmm-math/fixtures/helmholtz_angular.json;
//! tests 2 and 6 of the plan).

use nd_fmm_math::orthonormal::{DerivativeScratch, harmonics_derivatives};
use nd_fmm_math::{CONVENTION_VERSION, Layout};
use serde_json::Value;

use crate::rotation::{full, y};

fn fixture() -> Value {
    let text = include_str!("../../fixtures/helmholtz_angular.json");
    let doc: Value = serde_json::from_str(text).expect("the fixture is JSON");
    assert_eq!(
        doc["header"]["convention_version"].as_u64(),
        Some(CONVENTION_VERSION as u64),
        "regenerate the fixture: tools/fixtures/README.md"
    );
    doc
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|s| {
            s.as_str()
                .expect("a decimal string")
                .parse()
                .expect("an f64")
        })
        .collect()
}

fn integers(v: &Value) -> Vec<isize> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|i| i.as_i64().expect("an integer") as isize)
        .collect()
}

/// Test 2: Y, ∂θ Y and (1/sin θ) ∂φ Y at every sampled slot, degrees to 1000, seven
/// directions; each quantity relative to its largest sampled value (or the degree's
/// largest computed value where every sampled value is zero).
#[test]
fn harmonics_and_derivatives_agree_with_the_fixture() {
    let doc = fixture();
    let mut worst = [0.0f64; 3];
    for set in doc["harmonics"].as_array().expect("harmonic sets") {
        let n = set["n"].as_u64().expect("n") as usize;
        let x: Vec<f64> = numbers(&set["x"]);
        let layout = Layout::new(n);
        let mut computed = [
            vec![0.0; layout.len()],
            vec![0.0; layout.len()],
            vec![0.0; layout.len()],
        ];
        let [v, t, f] = &mut computed;
        harmonics_derivatives(
            n,
            [x[0], x[1], x[2]],
            v,
            t,
            f,
            &mut DerivativeScratch::new(n),
        );
        let slots = integers(&set["m"]);
        for (k, key) in ["value", "dtheta", "dphi"].iter().enumerate() {
            let expected = numbers(&set[key]);
            // The largest sampled value of the quantity, or, where every sampled value is
            // zero (∂θ at the equator for n + m even), the degree's largest computed one.
            let sampled = expected.iter().fold(0.0f64, |m, e| m.max(e.abs()));
            let scale = if sampled > 0.0 {
                sampled
            } else {
                computed[k][layout.degree(n)]
                    .iter()
                    .fold(0.0f64, |m, e| m.max(e.abs()))
                    .max(f64::MIN_POSITIVE)
            };
            let error = slots.iter().zip(&expected).fold(0.0f64, |m, (&slot, e)| {
                m.max((computed[k][layout.idx(n, slot)] - e).abs() / scale)
            });
            if error > 1e-14 {
                println!("{} n = {n} {key}: {error:.2e}", set["direction"]);
            }
            worst[k] = worst[k].max(error);
        }
    }
    println!(
        "worst: value {:.2e}, ∂θ {:.2e}, ∂φ {:.2e}",
        worst[0], worst[1], worst[2]
    );
    assert!(worst.iter().all(|&w| w <= 1e-13), "worst {worst:?}");
}

/// Test 6: every sampled block entry of degree ≤ `p`, absolute error.
fn blocks_agree(p: usize, select: impl Fn(usize) -> bool) -> f64 {
    let doc = fixture();
    let mut worst = 0.0f64;
    let mut cache: Option<(f64, Vec<f64>)> = None;
    for set in doc["rotation"].as_array().expect("rotation sets") {
        let n = set["n"].as_u64().expect("n") as usize;
        if !select(n) {
            continue;
        }
        let beta: f64 = set["beta"].as_str().expect("beta").parse().expect("an f64");
        if cache.as_ref().is_none_or(|(b, _)| *b != beta) {
            cache = Some((beta, y(p, beta)));
        }
        let block = full(n, &cache.as_ref().expect("cached").1);
        let w = 2 * n + 1;
        let (is, js, vs) = (
            integers(&set["i"]),
            integers(&set["j"]),
            numbers(&set["value"]),
        );
        let mut set_worst = 0.0f64;
        for ((i, j), v) in is.iter().zip(&js).zip(&vs) {
            let entry = block[(i + n as isize) as usize * w + (j + n as isize) as usize];
            set_worst = set_worst.max((entry - v).abs());
        }
        println!("n = {n:4} {:16} max |error| {set_worst:.2e}", set["angle"]);
        worst = worst.max(set_worst);
    }
    worst
}

/// Test 6: the fixture's blocks of degree ≤ 300 (sets ordered by degree, so one block
/// set per angle serves all).
#[test]
fn y_blocks_agree_with_the_fixture() {
    let worst = blocks_agree(300, |n| n <= 300);
    assert!(worst <= 1e-13, "worst {worst:e}");
}

/// Test 6 at large degree: the fixture's blocks of degree 500 to 1000.
#[test]
#[ignore = "release mode: block sets to degree 1000"]
fn y_blocks_agree_with_the_fixture_at_large_degree() {
    let worst = blocks_agree(1000, |n| n > 300);
    assert!(worst <= 1e-13, "worst {worst:e}");
}

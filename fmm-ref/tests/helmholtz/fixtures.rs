//! The direct sum against FMM3D's fixture fmm-ref/fixtures/helmholtz_fmm3d.json (Helmholtz
//! CH1.4, tools/fixtures/gen_helmholtz_fmm3d.py): `h3ddirectcg` with 200 sources and
//! 40 further targets, for real, complex and imaginary k and real and complex charges,
//! every value within 1.0e-15 of the kernel of §3.15 summed at 30 digits.
//!
//! Error measures: the relative ℓ2 error per case and quantity (proposal §9.1), and per
//! point relative to the term magnitudes (`common::magnitudes`). The fixture includes the
//! factor 1/(4π); the operators do not (§3.15, "Where the factors go"), so the fixture is
//! multiplied by 4π.

use nd_fmm_math::CONVENTION_VERSION;
use nd_fmm_ref::helmholtz::p2p;
use serde_json::Value;

use crate::common::{C, Worst, gradient_error, magnitudes, rel_l2, rel_l2_gradient};

/// Relative ℓ2 per case and quantity: the fixture is within 1.0e-15 of the exact sums
/// (CH1.4), measured here at most 1.0e-15, with a 10× margin for the last bits of other
/// libms.
const L2_TOL: f64 = 1e-14;
/// Per point relative to the term magnitudes: the fixture's own error is up to 2.7e-15 in
/// this measure (FMM3D at k = 25i), and the measured worst here 4.6e-15, so 5e-14.
const POINT_TOL: f64 = 5e-14;

struct Case {
    label: String,
    k: C,
    charges: Vec<C>,
    potential: Vec<C>,
    gradient: Vec<[C; 3]>,
}

struct Fixture {
    sources: Vec<[f64; 3]>,
    /// The sources, then the targets: the points the values are given at.
    points: Vec<[f64; 3]>,
    cases: Vec<Case>,
}

fn number(v: &Value) -> f64 {
    v.as_str()
        .expect("fixture numbers are decimal strings")
        .parse()
        .expect("fixture numbers parse as f64")
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array().expect("an array").iter().map(number).collect()
}

fn complex(v: &Value) -> C {
    let [re, im]: [f64; 2] = numbers(v).try_into().expect("[re, im]");
    C::new(re, im)
}

fn triples(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .expect("a list")
        .iter()
        .map(|p| numbers(p).try_into().expect("three coordinates"))
        .collect()
}

fn fixture() -> Fixture {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/helmholtz_fmm3d.json");
    let text = std::fs::read_to_string(path).expect("fmm-ref/fixtures/helmholtz_fmm3d.json");
    let v: Value = serde_json::from_str(&text).expect("the fixture is valid JSON");
    assert_eq!(
        v["header"]["convention_version"].as_u64(),
        Some(u64::from(CONVENTION_VERSION)),
        "the Helmholtz fixture was generated for another CONVENTION_VERSION"
    );
    let sources = triples(&v["sources"]);
    let mut points = sources.clone();
    points.extend(triples(&v["targets"]));
    let four_pi = 4.0 * std::f64::consts::PI;
    let cases = v["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| {
            let which = c["charges"].as_str().expect("real or complex");
            let charges = v["charges"][which]
                .as_array()
                .expect("charges")
                .iter()
                .map(complex)
                .collect();
            let gradient = c["gradient"]
                .as_array()
                .expect("gradient")
                .iter()
                .map(|g| {
                    let g = numbers(g);
                    core::array::from_fn(|i| C::new(g[2 * i], g[2 * i + 1]) * four_pi)
                })
                .collect();
            Case {
                label: c["label"].as_str().expect("label").to_owned(),
                k: complex(&c["k"]),
                charges,
                potential: c["potential"]
                    .as_array()
                    .expect("potential")
                    .iter()
                    .map(|p| complex(p) * four_pi)
                    .collect(),
                gradient,
            }
        })
        .collect();
    Fixture {
        sources,
        points,
        cases,
    }
}

/// Checks one evaluation of every case against the fixture.
fn check(
    name: &str,
    evaluate: impl Fn(C, &[[f64; 3]], &[C], &[[f64; 3]], &mut [C], &mut [[C; 3]]),
) {
    let f = fixture();
    assert_eq!(f.cases.len(), 12, "the fixture's twelve cases");
    let mut l2 = Worst::default();
    let mut point = Worst::default();
    for case in &f.cases {
        let n = f.points.len();
        assert_eq!(case.potential.len(), n);
        let mut potential = vec![C::new(0.0, 0.0); n];
        let mut gradient = vec![[C::new(0.0, 0.0); 3]; n];
        evaluate(
            case.k,
            &f.sources,
            &case.charges,
            &f.points,
            &mut potential,
            &mut gradient,
        );
        let e = rel_l2(&potential, &case.potential);
        l2.record(e, || format!("{}, potential", case.label));
        let e = rel_l2_gradient(&gradient, &case.gradient);
        l2.record(e, || format!("{}, gradient", case.label));
        for (i, &x) in f.points.iter().enumerate() {
            let (mp, mg) = magnitudes(case.k, &f.sources, &case.charges, x);
            let e = (potential[i] - case.potential[i]).norm() / mp;
            point.record(e, || format!("{}, potential at point {i}", case.label));
            let e = gradient_error(gradient[i], case.gradient[i]) / mg;
            point.record(e, || format!("{}, gradient at point {i}", case.label));
        }
    }
    l2.check(L2_TOL, &format!("{name}: relative ℓ2"));
    point.check(POINT_TOL, &format!("{name}: per point by term sum"));
}

#[test]
fn direct_sum_matches_fmm3d() {
    check("direct_sum", |k, s, q, t, phi, g| {
        p2p::direct_sum(k, s, q, t, phi, Some(g))
    });
}

#[test]
fn p2p_matches_fmm3d() {
    check("p2p", |k, s, q, t, phi, g| {
        p2p::p2p(k, s, q, t, phi, Some(g))
    });
}

#[test]
fn potential_alone_is_the_same() {
    // Without a gradient the potential is the same, bit for bit.
    let f = fixture();
    for case in &f.cases {
        let n = f.points.len();
        let mut with = vec![C::new(0.0, 0.0); n];
        let mut gradient = vec![[C::new(0.0, 0.0); 3]; n];
        p2p::direct_sum(
            case.k,
            &f.sources,
            &case.charges,
            &f.points,
            &mut with,
            Some(&mut gradient),
        );
        let mut without = vec![C::new(0.0, 0.0); n];
        p2p::direct_sum(
            case.k,
            &f.sources,
            &case.charges,
            &f.points,
            &mut without,
            None,
        );
        assert_eq!(with, without, "{}", case.label);
        let mut plain = vec![C::new(0.0, 0.0); n];
        p2p::p2p(
            case.k,
            &f.sources,
            &case.charges,
            &f.points,
            &mut plain,
            None,
        );
        let mut plain_with = vec![C::new(0.0, 0.0); n];
        p2p::p2p(
            case.k,
            &f.sources,
            &case.charges,
            &f.points,
            &mut plain_with,
            Some(&mut gradient),
        );
        assert_eq!(plain, plain_with, "{}", case.label);
    }
}

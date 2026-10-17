//! The dipole operators against the mpmath fixture fmm-ref/fixtures/dipoles.json
//! (tools/fixtures/gen_dipoles.py, 40 verified digits, p = 16): P2M and P2L coefficients in
//! three frames, and the dipole direct sum with its gradient.
//!
//! Error measures: coefficients per degree, relative to the largest fixture magnitude of
//! the degree ([`degree_error`]); potentials and gradients per target, relative to the sum
//! of the term magnitudes Σⱼ |dⱼ| / rⱼ² (potential) and Σⱼ 2 |dⱼ| / rⱼ³ (gradient, the
//! Euclidean norm), which bound the terms.

use nd_fmm_math::{CONVENTION_VERSION, RealScalar};
use nd_fmm_ref::{Frame, Workspace, leaf, p2p};
use serde_json::Value;

use crate::common::{degree_error, len, norm};

/// f64 coefficients against the fixture, per degree.
const COEFFICIENT_TOL: f64 = 1e-13;
/// f32 coefficients at p ≤ 8, per degree; the inputs are rounded to f32 first.
const COEFFICIENT_TOL_F32: f64 = 1e-5;
/// f64 potentials and gradients, relative to the term magnitudes: the compensated
/// `direct_sum_dipoles` and the plain `p2p_dipoles`.
const DIRECT_TOL: f64 = 1e-15;
const P2P_TOL: f64 = 1e-14;
/// f32 `p2p_dipoles`, relative to the term magnitudes.
const P2P_TOL_F32: f64 = 1e-5;

fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/dipoles.json");
    let text = std::fs::read_to_string(path).expect("fmm-ref/fixtures/dipoles.json is readable");
    let v: Value = serde_json::from_str(&text).expect("the fixture is valid JSON");
    assert_eq!(
        v["header"]["convention_version"].as_u64(),
        Some(u64::from(CONVENTION_VERSION)),
        "the dipole fixture was generated for another CONVENTION_VERSION"
    );
    v
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

fn triple(v: &Value) -> [f64; 3] {
    numbers(v).try_into().expect("three components")
}

fn triples(v: &Value) -> Vec<[f64; 3]> {
    v.as_array().expect("a list").iter().map(triple).collect()
}

/// One expansion case of the fixture.
struct Case {
    frame: Frame<f64>,
    sources: Vec<[f64; 3]>,
    dipoles: Vec<[f64; 3]>,
    coefficients: Vec<f64>,
}

fn cases(v: &Value) -> Vec<Case> {
    v.as_array()
        .expect("a list of cases")
        .iter()
        .map(|c| Case {
            frame: Frame::new(triple(&c["centre"]), number(&c["radius"])),
            sources: triples(&c["sources"]),
            dipoles: triples(&c["dipoles"]),
            coefficients: numbers(&c["coefficients"]),
        })
        .collect()
}

type Operator<T> = fn(usize, &Frame<T>, &[[T; 3]], &[[T; 3]], &mut Workspace<T>, &mut [T]);

/// The operator in T at degree p, as f64 coefficients.
fn apply<T: RealScalar>(op: Operator<T>, p: usize, case: &Case) -> Vec<f64> {
    let cast = |v: &[[f64; 3]]| -> Vec<[T; 3]> { v.iter().map(|x| x.map(T::from_f64)).collect() };
    let frame = Frame::new(
        case.frame.centre.map(T::from_f64),
        T::from_f64(case.frame.radius),
    );
    let mut ws = Workspace::new(p);
    let mut out = vec![T::zero(); len(p)];
    op(
        p,
        &frame,
        &cast(&case.sources),
        &cast(&case.dipoles),
        &mut ws,
        &mut out,
    );
    out.iter().map(|&x| RealScalar::to_f64(x)).collect()
}

fn check_expansions(name: &str, f64_op: Operator<f64>, f32_op: Operator<f32>) {
    let v = fixture();
    let p_max = v["header"]["p"].as_u64().unwrap() as usize;
    let (mut worst, mut worst32) = (0.0f64, 0.0f64);
    for (k, case) in cases(&v[name]).iter().enumerate() {
        for p in [0, 1, 3, 8, p_max] {
            let want = &case.coefficients[..len(p)];
            let e = degree_error(p, &apply(f64_op, p, case), want);
            assert!(
                e <= COEFFICIENT_TOL,
                "{name} case {k}, p = {p}: error {e:.3e}"
            );
            worst = worst.max(e);
        }
        // f32 only where the rounded inputs keep the frame: not the deep leaf far from
        // the origin, whose points round to the centre in f32.
        if case.frame.radius >= 0.25 {
            for p in [1, 3, 8] {
                let e = degree_error(p, &apply(f32_op, p, case), &case.coefficients[..len(p)]);
                assert!(
                    e <= COEFFICIENT_TOL_F32,
                    "{name} f32 case {k}, p = {p}: error {e:.3e}"
                );
                worst32 = worst32.max(e);
            }
        }
    }
    println!("{name}: worst per-degree error {worst:.2e} (f64), {worst32:.2e} (f32)");
}

#[test]
fn p2m_dipoles_matches_fixture() {
    check_expansions("p2m", leaf::p2m_dipoles::<f64>, leaf::p2m_dipoles::<f32>);
}

#[test]
fn p2l_dipoles_matches_fixture() {
    check_expansions("p2l", leaf::p2l_dipoles::<f64>, leaf::p2l_dipoles::<f32>);
}

#[test]
fn dipole_degree_zero_vanishes() {
    // Error measure: exact equality; ∇R₀⁰ = 0 (CONVENTIONS §3.7, "Dipole sources").
    let v = fixture();
    for case in cases(&v["p2m"]) {
        assert_eq!(apply(leaf::p2m_dipoles::<f64>, 4, &case)[0], 0.0);
    }
}

/// Σⱼ |dⱼ| / rⱼ² and Σⱼ 2 |dⱼ| / rⱼ³ at x.
fn magnitudes(sources: &[[f64; 3]], dipoles: &[[f64; 3]], x: [f64; 3]) -> (f64, f64) {
    sources
        .iter()
        .zip(dipoles)
        .fold((0.0, 0.0), |(a, b), (y, d)| {
            let r = norm(core::array::from_fn(|k| x[k] - y[k]));
            (a + norm(*d) / (r * r), b + 2.0 * norm(*d) / (r * r * r))
        })
}

#[test]
fn dipole_sums_match_fixture() {
    let v = fixture();
    let direct = &v["direct"];
    let sources = triples(&direct["sources"]);
    let dipoles = triples(&direct["dipoles"]);
    let records = direct["targets"].as_array().unwrap();
    let targets: Vec<[f64; 3]> = records.iter().map(|t| triple(&t["x"])).collect();
    let want_phi: Vec<f64> = records.iter().map(|t| number(&t["potential"])).collect();
    let want_grad: Vec<[f64; 3]> = records.iter().map(|t| triple(&t["gradient"])).collect();
    let n = targets.len();

    let check = |phi: &[f64], grad: &[[f64; 3]], tol: f64, what: &str| {
        for i in 0..n {
            let (sp, sg) = magnitudes(&sources, &dipoles, targets[i]);
            let ep = (phi[i] - want_phi[i]).abs() / sp;
            let eg = norm(core::array::from_fn(|k| grad[i][k] - want_grad[i][k])) / sg;
            assert!(
                ep <= tol && eg <= tol,
                "{what}, target {i}: errors {ep:.3e}, {eg:.3e}"
            );
        }
    };

    let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
    p2p::direct_sum_dipoles(&sources, &dipoles, &targets, &mut phi, Some(&mut grad));
    check(&phi, &grad, DIRECT_TOL, "direct_sum_dipoles");
    let mut only = vec![0.0; n];
    p2p::direct_sum_dipoles(&sources, &dipoles, &targets, &mut only, None);
    assert_eq!(only, phi, "the potential does not depend on the gradient");

    let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
    p2p::p2p_dipoles(&sources, &dipoles, &targets, &mut phi, Some(&mut grad));
    check(&phi, &grad, P2P_TOL, "p2p_dipoles");

    let cast =
        |v: &[[f64; 3]]| -> Vec<[f32; 3]> { v.iter().map(|x| x.map(|c| c as f32)).collect() };
    let (mut phi32, mut grad32) = (vec![0.0f32; n], vec![[0.0f32; 3]; n]);
    p2p::p2p_dipoles(
        &cast(&sources),
        &cast(&dipoles),
        &cast(&targets),
        &mut phi32,
        Some(&mut grad32),
    );
    let phi: Vec<f64> = phi32.iter().map(|&x| x as f64).collect();
    let grad: Vec<[f64; 3]> = grad32.iter().map(|g| g.map(|c| c as f64)).collect();
    check(&phi, &grad, P2P_TOL_F32, "p2p_dipoles f32");
}

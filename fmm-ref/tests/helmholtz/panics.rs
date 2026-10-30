//! Argument checks: each kind of length mismatch and a non-finite wavenumber panic.

use nd_fmm_ref::helmholtz::p2p;

use crate::common::C;

const K: C = C::new(1.0, 0.5);
const ONE: [C; 1] = [C::new(1.0, 0.0)];
const X: [[f64; 3]; 1] = [[1.0, 2.0, 3.0]];

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn p2p_charges_length() {
    p2p::p2p(K, &X, &[ONE[0], ONE[0]], &X, &mut [C::new(0.0, 0.0)], None);
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn p2p_potential_length() {
    p2p::p2p(K, &X, &ONE, &X, &mut [], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn p2p_gradient_length() {
    let mut g = [[C::new(0.0, 0.0); 3]; 2];
    p2p::p2p(K, &X, &ONE, &X, &mut [C::new(0.0, 0.0)], Some(&mut g));
}

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn direct_sum_charges_length() {
    p2p::direct_sum(K, &X, &[], &X, &mut [C::new(0.0, 0.0)], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn direct_sum_gradient_length() {
    let mut g: [[C; 3]; 0] = [];
    p2p::direct_sum(K, &X, &ONE, &X, &mut [C::new(0.0, 0.0)], Some(&mut g));
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn direct_sum_potential_length() {
    p2p::direct_sum(K, &X, &ONE, &X, &mut [], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn yukawa_gradient_length() {
    let mut g = [[0.0; 3]; 2];
    p2p::p2p_yukawa(1.0, &X, &[1.0], &X, &mut [0.0], Some(&mut g));
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn yukawa_potential_length() {
    p2p::p2p_yukawa(1.0, &X, &[1.0], &X, &mut [0.0, 0.0], None);
}

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn direct_sum_yukawa_charges_length() {
    p2p::direct_sum_yukawa(1.0, &X, &[], &X, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn p2p_nan_k() {
    p2p::p2p(
        C::new(f64::NAN, 0.0),
        &X,
        &ONE,
        &X,
        &mut [C::new(0.0, 0.0)],
        None,
    );
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn direct_sum_infinite_k() {
    p2p::direct_sum(
        C::new(1.0, f64::INFINITY),
        &X,
        &ONE,
        &X,
        &mut [C::new(0.0, 0.0)],
        None,
    );
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn yukawa_nan_kappa() {
    p2p::p2p_yukawa(f64::NAN, &X, &[1.0], &X, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn direct_sum_yukawa_infinite_kappa() {
    p2p::direct_sum_yukawa(f64::INFINITY, &X, &[1.0], &X, &mut [0.0], None);
}

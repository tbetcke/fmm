//! The dipole operators panic on mismatched lengths, with the messages of the charge
//! operators.

use nd_fmm_ref::{Frame, Workspace, leaf, p2p};

fn unit() -> Frame<f64> {
    Frame::new([0.0; 3], 1.0)
}

#[test]
#[should_panic(expected = "`sources` and `dipoles` must have the same length, got 1 and 2")]
fn p2m_dipoles_rejects_mismatched_moments() {
    let mut ws = Workspace::new(2);
    let mut out = vec![0.0; 9];
    leaf::p2m_dipoles(2, &unit(), &[[0.1; 3]], &[[1.0; 3]; 2], &mut ws, &mut out);
}

#[test]
#[should_panic(expected = "`local` must have length (p + 1)^2 = 9 for p = 2")]
fn p2l_dipoles_rejects_wrong_coefficients() {
    let mut ws = Workspace::new(2);
    let mut out = vec![0.0; 4];
    leaf::p2l_dipoles(2, &unit(), &[[3.0; 3]], &[[1.0; 3]], &mut ws, &mut out);
}

#[test]
#[should_panic(expected = "Workspace built for p = 1 is too small for p = 2")]
fn p2m_dipoles_rejects_small_workspace() {
    let mut ws = Workspace::new(1);
    let mut out = vec![0.0; 9];
    leaf::p2m_dipoles(2, &unit(), &[[0.1; 3]], &[[1.0; 3]], &mut ws, &mut out);
}

#[test]
#[should_panic(expected = "`sources` and `dipoles` must have the same length, got 2 and 1")]
fn p2p_dipoles_rejects_mismatched_moments() {
    let mut phi = [0.0];
    p2p::p2p_dipoles(&[[0.0; 3]; 2], &[[1.0; 3]], &[[1.0; 3]], &mut phi, None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 2")]
fn direct_sum_dipoles_rejects_wrong_gradient() {
    let (mut phi, mut grad) = ([0.0], [[0.0; 3]; 2]);
    p2p::direct_sum_dipoles(
        &[[0.0; 3]],
        &[[1.0; 3]],
        &[[1.0; 3]],
        &mut phi,
        Some(&mut grad),
    );
}

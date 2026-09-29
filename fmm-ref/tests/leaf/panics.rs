//! Argument checks: every operator panics on a workspace built for a smaller degree and
//! on slices of the wrong length.

use nd_fmm_ref::{Frame, Workspace, leaf};

fn frame() -> Frame<f64> {
    Frame::new([0.5, -0.5, 1.0], 0.37)
}

const NEAR: [[f64; 3]; 1] = [[0.6, -0.4, 1.1]];
const FAR: [[f64; 3]; 1] = [[2.0, 1.0, 1.0]];

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn p2m_rejects_small_workspace() {
    let mut out = [0.0; 25];
    leaf::p2m(4, &frame(), &NEAR, &[1.0], &mut Workspace::new(3), &mut out);
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn p2l_rejects_small_workspace() {
    let mut out = [0.0; 25];
    leaf::p2l(4, &frame(), &FAR, &[1.0], &mut Workspace::new(3), &mut out);
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn l2p_rejects_small_workspace() {
    let mut ws = Workspace::new(3);
    leaf::l2p(4, &frame(), &[0.0; 25], &NEAR, &mut ws, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn m2p_with_gradient_rejects_small_workspace() {
    let mut ws = Workspace::new(3);
    let mut gradient = [[0.0; 3]];
    let g = Some(&mut gradient[..]);
    leaf::m2p(4, &frame(), &[0.0; 25], &FAR, &mut ws, &mut [0.0], g);
}

#[test]
#[should_panic(expected = "`multipole` must have length (p + 1)^2 = 16 for p = 3")]
fn p2m_rejects_wrong_coefficient_length() {
    let mut out = [0.0; 25];
    leaf::p2m(3, &frame(), &NEAR, &[1.0], &mut Workspace::new(4), &mut out);
}

#[test]
#[should_panic(expected = "`local` must have length (p + 1)^2 = 16 for p = 3")]
fn l2p_rejects_wrong_coefficient_length() {
    let mut ws = Workspace::new(3);
    leaf::l2p(3, &frame(), &[0.0; 9], &NEAR, &mut ws, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 1 and 2")]
fn p2l_rejects_mismatched_charges() {
    let mut out = [0.0; 4];
    leaf::p2l(
        1,
        &frame(),
        &FAR,
        &[1.0, 2.0],
        &mut Workspace::new(1),
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`targets` and `potential` must have the same length, got 1 and 2")]
fn m2p_rejects_mismatched_potential() {
    let mut ws = Workspace::new(1);
    leaf::m2p(1, &frame(), &[0.0; 4], &FAR, &mut ws, &mut [0.0; 2], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 0")]
fn l2p_rejects_mismatched_gradient() {
    let mut ws = Workspace::new(1);
    let mut empty: [[f64; 3]; 0] = [];
    let g = Some(&mut empty[..]);
    leaf::l2p(1, &frame(), &[0.0; 4], &NEAR, &mut ws, &mut [0.0], g);
}

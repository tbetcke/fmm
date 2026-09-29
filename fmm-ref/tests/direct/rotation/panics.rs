//! Argument checks: every rotation-based translation panics, with the message of its
//! `direct` counterpart, on a workspace built for a smaller degree and on coefficient
//! slices of the wrong length.

use nd_fmm_ref::{Frame, Workspace, rotation};

fn source() -> Frame<f64> {
    Frame::new([0.5, -0.5, 1.0], 0.37)
}

fn target() -> Frame<f64> {
    Frame::new([2.0, -0.5, 1.5], 0.37)
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn m2m_rejects_small_workspace() {
    let mut out = [0.0; 25];
    let mut ws = Workspace::new(3);
    rotation::m2m(4, &source(), &target(), &mut ws, &[0.0; 25], &mut out);
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn l2l_rejects_small_workspace() {
    let mut out = [0.0; 25];
    let mut ws = Workspace::new(3);
    rotation::l2l(4, &source(), &target(), &mut ws, &[0.0; 25], &mut out);
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn m2l_rejects_small_workspace() {
    let mut out = [0.0; 25];
    let mut ws = Workspace::new(3);
    rotation::m2l(4, &source(), &target(), &mut ws, &[0.0; 25], &mut out);
}

#[test]
#[should_panic(expected = "`multipole_in` must have length (p + 1)^2 = 16 for p = 3")]
fn m2m_rejects_short_input() {
    let mut out = [0.0; 16];
    let mut ws = Workspace::new(3);
    rotation::m2m(3, &source(), &target(), &mut ws, &[0.0; 9], &mut out);
}

#[test]
#[should_panic(expected = "`local_out` must have length (p + 1)^2 = 16 for p = 3")]
fn l2l_rejects_short_output() {
    let mut out = [0.0; 9];
    let mut ws = Workspace::new(3);
    rotation::l2l(3, &source(), &target(), &mut ws, &[0.0; 16], &mut out);
}

#[test]
#[should_panic(expected = "`local` must have length (p + 1)^2 = 16 for p = 3")]
fn m2l_rejects_long_output() {
    let mut out = [0.0; 25];
    let mut ws = Workspace::new(3);
    rotation::m2l(3, &source(), &target(), &mut ws, &[0.0; 16], &mut out);
}

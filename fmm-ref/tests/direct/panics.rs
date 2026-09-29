//! Argument checks: every translation panics on a workspace built for a smaller degree
//! and on coefficient slices of the wrong length.

use nd_fmm_ref::{Frame, Workspace, direct};

fn source() -> Frame<f64> {
    Frame::new([0.5, -0.5, 1.0], 0.37)
}

fn target() -> Frame<f64> {
    Frame::new([2.0, -0.5, 1.0], 0.37)
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn m2m_rejects_small_workspace() {
    let mut out = [0.0; 25];
    direct::m2m(
        4,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 25],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn l2l_rejects_small_workspace() {
    let mut out = [0.0; 25];
    direct::l2l(
        4,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 25],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "Workspace built for p = 3 is too small for p = 4")]
fn m2l_rejects_small_workspace() {
    let mut out = [0.0; 25];
    direct::m2l(
        4,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 25],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`multipole_in` must have length (p + 1)^2 = 16 for p = 3")]
fn m2m_rejects_short_input() {
    let mut out = [0.0; 16];
    direct::m2m(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 9],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`multipole_out` must have length (p + 1)^2 = 16 for p = 3")]
fn m2m_rejects_long_output() {
    let mut out = [0.0; 25];
    direct::m2m(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 16],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`local_in` must have length (p + 1)^2 = 16 for p = 3")]
fn l2l_rejects_long_input() {
    let mut out = [0.0; 16];
    direct::l2l(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 25],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`local_out` must have length (p + 1)^2 = 16 for p = 3")]
fn l2l_rejects_short_output() {
    let mut out = [0.0; 9];
    direct::l2l(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 16],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`multipole` must have length (p + 1)^2 = 16 for p = 3")]
fn m2l_rejects_short_input() {
    let mut out = [0.0; 16];
    direct::m2l(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 9],
        &mut out,
    );
}

#[test]
#[should_panic(expected = "`local` must have length (p + 1)^2 = 16 for p = 3")]
fn m2l_rejects_short_output() {
    let mut out = [0.0; 9];
    direct::m2l(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[0.0; 16],
        &mut out,
    );
}

#[test]
fn m2l_at_the_workspace_degree_fits() {
    // M2L needs shift harmonics up to degree 2p; a workspace built for p holds them.
    let mut out = [0.0; 16];
    direct::m2l(
        3,
        &source(),
        &target(),
        &mut Workspace::new(3),
        &[1.0; 16],
        &mut out,
    );
    assert!(out.iter().all(|c| c.is_finite()));
}

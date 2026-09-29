//! Argument checks: both functions panic on paired slices of different lengths.

use nd_fmm_ref::p2p::{direct_sum, p2p};

const SOURCES: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
const TARGETS: [[f64; 3]; 1] = [[0.0, 2.0, 0.0]];

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 2 and 1")]
fn p2p_rejects_mismatched_charges() {
    p2p(&SOURCES, &[1.0], &TARGETS, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "`targets` and `potential` must have the same length, got 1 and 2")]
fn p2p_rejects_mismatched_potential() {
    p2p(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0; 2], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 0")]
fn p2p_rejects_mismatched_gradient() {
    p2p(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0], Some(&mut []));
}

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 2 and 3")]
fn direct_sum_rejects_mismatched_charges() {
    direct_sum(&SOURCES, &[1.0; 3], &TARGETS, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "`targets` and `potential` must have the same length, got 1 and 0")]
fn direct_sum_rejects_mismatched_potential() {
    direct_sum(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 2")]
fn direct_sum_rejects_mismatched_gradient() {
    let mut gradient = [[0.0; 3]; 2];
    direct_sum(
        &SOURCES,
        &[1.0, 1.0],
        &TARGETS,
        &mut [0.0],
        Some(&mut gradient),
    );
}

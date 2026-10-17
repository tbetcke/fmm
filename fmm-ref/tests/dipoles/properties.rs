//! Accumulation, linearity in the moment, the coincident-pair rule and agreement of the
//! two dipole sums.

use nd_fmm_ref::{Frame, Workspace, leaf, p2p};

use crate::common::{SQRT3, SplitMix64, frames, len, place};

type Dipole = fn(usize, &Frame<f64>, &[[f64; 3]], &[[f64; 3]], &mut Workspace<f64>, &mut [f64]);

#[test]
fn expansions_accumulate() {
    // Error measure: exact equality. One source per call: the operator adds its terms
    // into the existing values, so adding into a filled output equals adding into zeros
    // and then onto the filled values, slot by slot.
    let mut rng = SplitMix64::new(21);
    let p = 8;
    let mut ws = Workspace::new(p);
    for frame in frames() {
        for (op, radii) in [
            (leaf::p2m_dipoles as Dipole, (0.0, SQRT3)),
            (leaf::p2l_dipoles, (2.0, 8.0)),
        ] {
            let y = [place(&frame, rng.in_shell(radii.0, radii.1))];
            let d = [rng.moment()];
            let filled: Vec<f64> = (0..len(p)).map(|_| rng.range(-1.0, 1.0)).collect();
            let mut fresh = vec![0.0; len(p)];
            op(p, &frame, &y, &d, &mut ws, &mut fresh);
            let mut into = filled.clone();
            op(p, &frame, &y, &d, &mut ws, &mut into);
            let want: Vec<f64> = filled.iter().zip(&fresh).map(|(a, b)| a + b).collect();
            assert_eq!(into, want);
        }
    }
}

#[test]
fn a_moment_along_an_axis_scales_exactly() {
    // Error measure: exact equality. A moment 2^k e_x scales every term by 2^k exactly
    // (the contraction multiplies by zero in the other components).
    let frame = Frame::new([0.25, -0.5, 0.75], 0.5);
    let p = 6;
    let mut ws = Workspace::new(p);
    let y = [place(&frame, [0.3, -0.4, 0.2])];
    let mut one = vec![0.0; len(p)];
    leaf::p2m_dipoles(p, &frame, &y, &[[1.0, 0.0, 0.0]], &mut ws, &mut one);
    let mut eight = vec![0.0; len(p)];
    leaf::p2m_dipoles(p, &frame, &y, &[[8.0, 0.0, 0.0]], &mut ws, &mut eight);
    assert_eq!(eight, one.iter().map(|c| 8.0 * c).collect::<Vec<_>>());
}

#[test]
fn coincident_pairs_are_skipped() {
    // Error measure: exact equality.
    let points = [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]];
    let dipoles = [[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]];
    let (mut phi, mut grad) = ([0.0; 2], [[0.0; 3]; 2]);
    p2p::p2p_dipoles(&points, &dipoles, &points, &mut phi, Some(&mut grad));
    // Target 0 sees only the dipole (0, 0, 3) at (1, 2, 2): w = (−1, −2, −2), r = 3,
    // s = −6, φ = −6/27; ∇φ = (d − 3 s w / r²) / r³ = ((0, 0, 3) − 2 (1, 2, 2)) / 27.
    assert_eq!(phi[0], -6.0 / 27.0);
    assert_eq!(grad[0], [-2.0 / 27.0, -4.0 / 27.0, -1.0 / 27.0]);
    // Target 1 sees only the dipole e_x at the origin: w = (1, 2, 2), s = 1, φ = 1/27.
    assert_eq!(phi[1], 1.0 / 27.0);
    let mut oracle = [0.0; 2];
    p2p::direct_sum_dipoles(&points, &dipoles, &points, &mut oracle, None);
    assert_eq!(oracle, phi);
}

#[test]
fn plain_and_compensated_sums_agree() {
    // Error measure: relative to the largest term magnitude Σ |d| / r² over the targets,
    // f64: both form the same terms; only the summation differs.
    let mut rng = SplitMix64::new(22);
    let sources: Vec<[f64; 3]> = (0..200).map(|_| rng.in_shell(0.0, 1.0)).collect();
    let dipoles: Vec<[f64; 3]> = (0..200).map(|_| rng.moment()).collect();
    let targets: Vec<[f64; 3]> = (0..20).map(|_| rng.in_shell(1.5, 3.0)).collect();
    let (mut a, mut b) = (vec![0.0; 20], vec![0.0; 20]);
    p2p::p2p_dipoles(&sources, &dipoles, &targets, &mut a, None);
    p2p::direct_sum_dipoles(&sources, &dipoles, &targets, &mut b, None);
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-13 * 200.0, "{x} against {y}");
    }
}

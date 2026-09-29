//! Linearity and accumulation: an operator applied to a set of sources (or to a sum of
//! expansions) equals the sum over the parts, and every operator adds onto nonzero
//! initial values.

use nd_fmm_ref::{Frame, Workspace, leaf};

use crate::common::{
    Basis, Kind, P_MAX, SplitMix64, Worst, degree_error, frames, len, norm, place, relative,
    term_magnitudes,
};

/// Tolerance of the per-degree coefficient comparisons, relative to the summed
/// weighted norms of the parts.
const COEFFICIENT_TOL: f64 = 1e-14;
/// Tolerance of the potential and gradient comparisons, relative to the summed term
/// magnitudes of the parts.
const EVALUATION_TOL: f64 = 1e-14;

/// A sampler of points in a frame.
type Source = fn(&Frame<f64>, &mut SplitMix64) -> [f64; 3];

/// The signature of P2M and P2L.
type Expand = fn(usize, &Frame<f64>, &[[f64; 3]], &[f64], &mut Workspace<f64>, &mut [f64]);

/// The signature of L2P and M2P.
type Evaluate = fn(
    usize,
    &Frame<f64>,
    &[f64],
    &[[f64; 3]],
    &mut Workspace<f64>,
    &mut [f64],
    Option<&mut [[f64; 3]]>,
);

/// P2M or P2L with its kind and a sampler of sources in the tested range.
fn expansions() -> [(Kind, Expand, Source); 2] {
    [
        (Kind::Multipole, leaf::p2m, |f, rng| place(f, rng.in_cube())),
        (Kind::Local, leaf::p2l, |f, rng| {
            place(f, rng.in_shell(2.0, 6.0))
        }),
    ]
}

#[test]
fn p2m_and_p2l_are_additive_over_sources() {
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for P2M,
    // Nₘ/Sₘ for P2L), relative to the sum of the weighted norms of the two parts.
    let mut rng = SplitMix64::new(0x11e1);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = Worst::new("P2M/P2L additivity, per degree");
    for (kind, expand, source) in expansions() {
        for frame in frames() {
            let sources: Vec<[f64; 3]> = (0..12).map(|_| source(&frame, &mut rng)).collect();
            let charges: Vec<f64> = (0..12).map(|_| rng.charge()).collect();
            for p in [0, 1, 4, 13, P_MAX] {
                let mut whole = vec![0.0; len(p)];
                expand(p, &frame, &sources, &charges, &mut ws, &mut whole);
                let (mut a, mut b) = (vec![0.0; len(p)], vec![0.0; len(p)]);
                expand(p, &frame, &sources[..5], &charges[..5], &mut ws, &mut a);
                expand(p, &frame, &sources[5..], &charges[5..], &mut ws, &mut b);
                let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                let e = worst.update(degree_error(kind, p, &whole, &sum, &[&a, &b]));
                assert!(e <= COEFFICIENT_TOL, "{kind:?} p = {p}: {e:.3e}");

                // Doubling every charge doubles every coefficient exactly.
                let doubled: Vec<f64> = charges.iter().map(|q| 2.0 * q).collect();
                let mut twice = vec![0.0; len(p)];
                expand(p, &frame, &sources, &doubled, &mut ws, &mut twice);
                let want: Vec<f64> = whole.iter().map(|c| 2.0 * c).collect();
                assert_eq!(twice, want, "{kind:?} p = {p}: charge scaling");
            }
        }
    }
}

#[test]
fn p2m_and_p2l_accumulate_onto_initial_values() {
    // Error measure: as above, relative to the weighted norms of the initial values
    // and of the fresh result. Also checks that no sources leave the output untouched.
    let mut rng = SplitMix64::new(0x11e2);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = Worst::new("P2M/P2L accumulation, per degree");
    for (kind, expand, source) in expansions() {
        for frame in frames() {
            let sources: Vec<[f64; 3]> = (0..6).map(|_| source(&frame, &mut rng)).collect();
            let charges: Vec<f64> = (0..6).map(|_| rng.charge()).collect();
            for p in [0, 3, 17, P_MAX] {
                // Initial values of the size of a real expansion: another expansion.
                let mut initial = vec![0.0; len(p)];
                let other: Vec<[f64; 3]> = (0..3).map(|_| source(&frame, &mut rng)).collect();
                expand(p, &frame, &other, &[0.5, -1.0, 0.75], &mut ws, &mut initial);

                let mut fresh = vec![0.0; len(p)];
                expand(p, &frame, &sources, &charges, &mut ws, &mut fresh);
                let mut accumulated = initial.clone();
                expand(p, &frame, &sources, &charges, &mut ws, &mut accumulated);
                let want: Vec<f64> = initial.iter().zip(&fresh).map(|(x, y)| x + y).collect();
                let e = degree_error(kind, p, &accumulated, &want, &[&initial, &fresh]);
                assert!(
                    worst.update(e) <= COEFFICIENT_TOL,
                    "{kind:?} p = {p}: {e:.3e}"
                );

                let mut untouched = initial.clone();
                expand(p, &frame, &[], &[], &mut ws, &mut untouched);
                assert_eq!(untouched, initial);
            }
        }
    }
}

/// L2P and M2P with the kind of expansion they read, the matching P2L or P2M, the
/// basis they evaluate, a sampler of sources and one of targets in the tested ranges.
fn evaluations() -> [(Evaluate, Expand, Basis, Source, Source); 2] {
    [
        (
            leaf::l2p,
            leaf::p2l,
            Basis::Regular,
            |f, rng| place(f, rng.in_shell(2.0, 6.0)),
            |f, rng| place(f, rng.in_cube()),
        ),
        (
            leaf::m2p,
            leaf::p2m,
            Basis::Irregular,
            |f, rng| place(f, rng.in_cube()),
            |f, rng| place(f, rng.in_shell(2.0, 6.0)),
        ),
    ]
}

#[test]
fn l2p_and_m2p_are_linear_in_the_expansion() {
    // Error measure: potential relative to the summed term magnitudes of both
    // expansions at the target, gradient (Euclidean) relative to the summed term
    // gradient norms (`term_magnitudes`).
    let mut rng = SplitMix64::new(0x11e3);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = Worst::new("L2P/M2P linearity in the expansion");
    for (evaluate, expand, basis, source, target) in evaluations() {
        for frame in frames() {
            let targets: Vec<[f64; 3]> = (0..8).map(|_| target(&frame, &mut rng)).collect();
            for p in [0, 1, 6, 19, P_MAX] {
                let mut parts = [vec![0.0; len(p)], vec![0.0; len(p)]];
                for part in &mut parts {
                    let sources: Vec<[f64; 3]> = (0..4).map(|_| source(&frame, &mut rng)).collect();
                    let charges: Vec<f64> = (0..4).map(|_| rng.charge()).collect();
                    expand(p, &frame, &sources, &charges, &mut ws, part);
                }
                let sum: Vec<f64> = parts[0].iter().zip(&parts[1]).map(|(a, b)| a + b).collect();
                let n = targets.len();
                let mut outputs = [0; 3].map(|_| (vec![0.0; n], vec![[0.0; 3]; n]));
                for ((phi, grad), coefficients) in
                    outputs.iter_mut().zip([&sum, &parts[0], &parts[1]])
                {
                    evaluate(p, &frame, coefficients, &targets, &mut ws, phi, Some(grad));
                }
                for (i, &x) in targets.iter().enumerate() {
                    let (v0, g0) = term_magnitudes(basis, p, &frame, &parts[0], x);
                    let (v1, g1) = term_magnitudes(basis, p, &frame, &parts[1], x);
                    let value = outputs[1].0[i] + outputs[2].0[i];
                    let e = relative((outputs[0].0[i] - value).abs(), v0 + v1);
                    assert!(
                        worst.update(e) <= EVALUATION_TOL,
                        "{basis:?} p = {p}: {e:.3e}"
                    );
                    let diff: [f64; 3] = core::array::from_fn(|k| {
                        outputs[0].1[i][k] - outputs[1].1[i][k] - outputs[2].1[i][k]
                    });
                    let e = relative(norm(diff), g0 + g1);
                    assert!(
                        worst.update(e) <= EVALUATION_TOL,
                        "{basis:?} p = {p}: {e:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn l2p_and_m2p_accumulate_and_treat_targets_independently() {
    // Error measure: exact equality. Each target receives one addition per output, so
    // evaluating onto initial values gives initial + fresh bit for bit, and splitting
    // the targets changes nothing. Without a gradient the potential is the same, bit
    // for bit.
    let mut rng = SplitMix64::new(0x11e4);
    let mut ws = Workspace::new(P_MAX);
    for (evaluate, expand, _, source, target) in evaluations() {
        for frame in frames() {
            let targets: Vec<[f64; 3]> = (0..9).map(|_| target(&frame, &mut rng)).collect();
            let sources: Vec<[f64; 3]> = (0..5).map(|_| source(&frame, &mut rng)).collect();
            let charges: Vec<f64> = (0..5).map(|_| rng.charge()).collect();
            for p in [0, 2, 11, P_MAX] {
                let mut coefficients = vec![0.0; len(p)];
                expand(p, &frame, &sources, &charges, &mut ws, &mut coefficients);
                let n = targets.len();

                let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
                evaluate(
                    p,
                    &frame,
                    &coefficients,
                    &targets,
                    &mut ws,
                    &mut phi,
                    Some(&mut grad),
                );

                let initial_phi: Vec<f64> = (0..n).map(|_| rng.range(-3.0, 3.0)).collect();
                let initial_grad: Vec<[f64; 3]> = (0..n)
                    .map(|_| [0; 3].map(|_| rng.range(-3.0, 3.0)))
                    .collect();
                let (mut acc_phi, mut acc_grad) = (initial_phi.clone(), initial_grad.clone());
                evaluate(
                    p,
                    &frame,
                    &coefficients,
                    &targets,
                    &mut ws,
                    &mut acc_phi,
                    Some(&mut acc_grad),
                );
                for i in 0..n {
                    assert_eq!(acc_phi[i], initial_phi[i] + phi[i]);
                    for k in 0..3 {
                        assert_eq!(acc_grad[i][k], initial_grad[i][k] + grad[i][k]);
                    }
                }

                let (mut split_phi, mut split_grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
                let (phi_a, phi_b) = split_phi.split_at_mut(4);
                let (grad_a, grad_b) = split_grad.split_at_mut(4);
                evaluate(
                    p,
                    &frame,
                    &coefficients,
                    &targets[..4],
                    &mut ws,
                    phi_a,
                    Some(grad_a),
                );
                evaluate(
                    p,
                    &frame,
                    &coefficients,
                    &targets[4..],
                    &mut ws,
                    phi_b,
                    Some(grad_b),
                );
                assert_eq!(split_phi, phi);
                assert_eq!(split_grad, grad);

                let mut phi_only = vec![0.0; n];
                evaluate(
                    p,
                    &frame,
                    &coefficients,
                    &targets,
                    &mut ws,
                    &mut phi_only,
                    None,
                );
                assert_eq!(phi_only, phi);
            }
        }
    }
}

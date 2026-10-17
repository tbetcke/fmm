//! Dipole P2M then M2P, and dipole P2L then L2P, against the dipole direct sum: the
//! expansions converge to the exact potential and gradient of the dipoles
//! (CONVENTIONS §3.7, "Dipole sources": the evaluation formulas apply unchanged).
//!
//! Sources at |u| ≤ √3 and targets at 4 ≤ |v| ≤ 8 (multipoles), and the reverse
//! (locals), in the test frames; four sources with moments in [−1, 1)³ r. The truncation
//! error at degree p is of order (p + 2) (√3 / 4)^(p + 1) relative (the charge estimate
//! with one more power of the degree for the derivative); the tolerance is twice that,
//! plus 10⁻¹³ for rounding.
//!
//! Error measure: per target, relative to the term magnitudes Σⱼ |dⱼ| / rⱼ² (potential)
//! and Σⱼ 2 |dⱼ| / rⱼ³ (gradient, Euclidean norm), f64.

use nd_fmm_ref::{Frame, Workspace, leaf, p2p};

use crate::common::{SQRT3, SplitMix64, frames, len, norm, place};

/// The degrees checked, each against [`tolerance`].
const DEGREES: [usize; 3] = [4, 12, 30];

/// 2 (p + 2) (√3 / 4)^(p + 1) + 10⁻¹³.
fn tolerance(p: usize) -> f64 {
    2.0 * (p as f64 + 2.0) * (SQRT3 / 4.0).powi(p as i32 + 1) + 1e-13
}

fn magnitudes(sources: &[[f64; 3]], dipoles: &[[f64; 3]], x: [f64; 3]) -> (f64, f64) {
    sources
        .iter()
        .zip(dipoles)
        .fold((0.0, 0.0), |(a, b), (y, d)| {
            let r = norm(core::array::from_fn(|k| x[k] - y[k]));
            (a + norm(*d) / (r * r), b + 2.0 * norm(*d) / (r * r * r))
        })
}

#[derive(Clone, Copy)]
enum Chain {
    Multipole,
    Local,
}

fn check(chain: Chain, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let (near, far) = ((0.0, SQRT3), (4.0, 8.0));
    let (source_radii, target_radii) = match chain {
        Chain::Multipole => (near, far),
        Chain::Local => (far, near),
    };
    for frame in frames() {
        let sources: Vec<[f64; 3]> = (0..4)
            .map(|_| place(&frame, rng.in_shell(source_radii.0, source_radii.1)))
            .collect();
        let dipoles: Vec<[f64; 3]> = (0..4)
            .map(|_| rng.moment().map(|c| c * frame.radius))
            .collect();
        let targets: Vec<[f64; 3]> = (0..6)
            .map(|_| place(&frame, rng.in_shell(target_radii.0, target_radii.1)))
            .collect();
        let n = targets.len();
        let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
        p2p::direct_sum_dipoles(&sources, &dipoles, &targets, &mut phi, Some(&mut grad));
        let mut previous = f64::INFINITY;
        for p in DEGREES {
            let tol = tolerance(p);
            let (got_phi, got_grad) = evaluate(chain, p, &frame, (&sources, &dipoles), &targets);
            let mut worst: f64 = 0.0;
            for i in 0..n {
                let (sp, sg) = magnitudes(&sources, &dipoles, targets[i]);
                let ep = (got_phi[i] - phi[i]).abs() / sp;
                let eg = norm(core::array::from_fn(|k| got_grad[i][k] - grad[i][k])) / sg;
                worst = worst.max(ep).max(eg);
            }
            assert!(
                worst <= tol,
                "frame radius {}, p = {p}: error {worst:.3e}",
                frame.radius
            );
            assert!(worst < previous, "the error does not decrease with p");
            previous = worst;
        }
    }
}

/// The chain at degree p: the potentials and gradients at `targets`.
fn evaluate(
    chain: Chain,
    p: usize,
    frame: &Frame<f64>,
    (sources, dipoles): (&[[f64; 3]], &[[f64; 3]]),
    targets: &[[f64; 3]],
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut ws = Workspace::new(p);
    let mut coefficients = vec![0.0; len(p)];
    let (mut phi, mut grad) = (vec![0.0; targets.len()], vec![[0.0; 3]; targets.len()]);
    match chain {
        Chain::Multipole => {
            leaf::p2m_dipoles(p, frame, sources, dipoles, &mut ws, &mut coefficients);
            leaf::m2p(
                p,
                frame,
                &coefficients,
                targets,
                &mut ws,
                &mut phi,
                Some(&mut grad),
            );
        }
        Chain::Local => {
            leaf::p2l_dipoles(p, frame, sources, dipoles, &mut ws, &mut coefficients);
            leaf::l2p(
                p,
                frame,
                &coefficients,
                targets,
                &mut ws,
                &mut phi,
                Some(&mut grad),
            );
        }
    }
    (phi, grad)
}

#[test]
fn dipole_multipole_converges_to_direct_sum() {
    check(Chain::Multipole, 11);
}

#[test]
fn dipole_local_converges_to_direct_sum() {
    check(Chain::Local, 12);
}

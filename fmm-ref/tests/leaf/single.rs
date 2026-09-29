//! f32 against f64 for p ≤ 8 (CONVENTIONS §3.9): each operator run in f32 matches the
//! same operator run in f64 on the same (f32-representable) inputs to 1e-5, measured
//! as in the corresponding f64 test.

use nd_fmm_ref::{Frame, Workspace, leaf};

use crate::common::{
    Basis, Kind, P_MAX_F32, SplitMix64, Worst, degree_error, frames, len, norm, place, relative,
    sub, term_magnitudes,
};

const TOL: f64 = 1e-5;

/// A sampler of scaled positions: sources first, then targets.
type Sampler = fn(&mut SplitMix64) -> [f64; 3];

/// The f32 twin of an f64 frame and the f64 frame of the rounded values.
fn twins(frame: &Frame<f64>) -> (Frame<f32>, Frame<f64>) {
    let single = Frame::new(frame.centre.map(|c| c as f32), frame.radius as f32);
    let double = Frame::new(single.centre.map(f64::from), f64::from(single.radius));
    (single, double)
}

fn to_f32(points: &[[f64; 3]]) -> Vec<[f32; 3]> {
    points.iter().map(|x| x.map(|c| c as f32)).collect()
}

fn to_f64<const N: usize>(values: &[[f32; N]]) -> Vec<[f64; N]> {
    values.iter().map(|x| x.map(f64::from)).collect()
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// Runs one expansion (P2M or P2L) and its evaluation (M2P or L2P) in both
/// precisions and checks each against its f64 twin, with sources and targets drawn
/// from `samplers` in scaled coordinates.
fn compare(kind: Kind, basis: Basis, samplers: [Sampler; 2], worst: &mut [Worst; 3]) {
    let mut rng = SplitMix64::new(0xf32 + kind as u64);
    let (mut ws32, mut ws64) = (Workspace::<f32>::new(P_MAX_F32), Workspace::new(P_MAX_F32));
    let [sample_source, sample_target] = samplers;
    for frame in frames() {
        let (frame32, frame64) = twins(&frame);
        let sources32 = to_f32(
            &(0..6)
                .map(|_| place(&frame, sample_source(&mut rng)))
                .collect::<Vec<_>>(),
        );
        let targets32 = to_f32(
            &(0..6)
                .map(|_| place(&frame, sample_target(&mut rng)))
                .collect::<Vec<_>>(),
        );
        let charges32: Vec<f32> = (0..6).map(|_| rng.charge() as f32).collect();
        let (sources64, targets64, charges64) =
            (to_f64(&sources32), to_f64(&targets32), widen(&charges32));
        for p in 0..=P_MAX_F32 {
            // Expansion: coefficients per degree, weighted as for the f64 tests.
            let (mut c32, mut c64) = (vec![0.0f32; len(p)], vec![0.0; len(p)]);
            match kind {
                Kind::Multipole => {
                    leaf::p2m(p, &frame32, &sources32, &charges32, &mut ws32, &mut c32);
                    leaf::p2m(p, &frame64, &sources64, &charges64, &mut ws64, &mut c64);
                }
                Kind::Local => {
                    leaf::p2l(p, &frame32, &sources32, &charges32, &mut ws32, &mut c32);
                    leaf::p2l(p, &frame64, &sources64, &charges64, &mut ws64, &mut c64);
                }
            }
            let e = worst[0].update(degree_error(kind, p, &widen(&c32), &c64, &[&c64]));
            assert!(e <= TOL, "{kind:?} p = {p}: coefficients {e:.3e}");

            // Evaluation of the f32 coefficients: potential relative to the sum of
            // term magnitudes, gradient relative to the sum of term gradient norms.
            let c32_as_64 = widen(&c32);
            let n = targets32.len();
            let (mut phi32, mut grad32) = (vec![0.0f32; n], vec![[0.0f32; 3]; n]);
            let (mut phi64, mut grad64) = (vec![0.0; n], vec![[0.0; 3]; n]);
            match basis {
                Basis::Regular => {
                    leaf::l2p(
                        p,
                        &frame32,
                        &c32,
                        &targets32,
                        &mut ws32,
                        &mut phi32,
                        Some(&mut grad32),
                    );
                    leaf::l2p(
                        p,
                        &frame64,
                        &c32_as_64,
                        &targets64,
                        &mut ws64,
                        &mut phi64,
                        Some(&mut grad64),
                    );
                }
                Basis::Irregular => {
                    leaf::m2p(
                        p,
                        &frame32,
                        &c32,
                        &targets32,
                        &mut ws32,
                        &mut phi32,
                        Some(&mut grad32),
                    );
                    leaf::m2p(
                        p,
                        &frame64,
                        &c32_as_64,
                        &targets64,
                        &mut ws64,
                        &mut phi64,
                        Some(&mut grad64),
                    );
                }
            }
            for i in 0..n {
                let (value_scale, gradient_scale) =
                    term_magnitudes(basis, p, &frame64, &c32_as_64, targets64[i]);
                let e = relative((f64::from(phi32[i]) - phi64[i]).abs(), value_scale);
                assert!(
                    worst[1].update(e) <= TOL,
                    "{basis:?} p = {p}: potential {e:.3e}"
                );
                let g32 = grad32[i].map(f64::from);
                let e = relative(norm(sub(g32, grad64[i])), gradient_scale);
                assert!(
                    worst[2].update(e) <= TOL,
                    "{basis:?} p = {p}: gradient {e:.3e}"
                );
            }
        }
    }
}

#[test]
fn p2m_and_m2p_in_f32_match_f64() {
    let mut worst = [
        Worst::new("f32 P2M vs f64, per degree"),
        Worst::new("f32 M2P potential vs f64"),
        Worst::new("f32 M2P gradient vs f64"),
    ];
    compare(
        Kind::Multipole,
        Basis::Irregular,
        [|rng| rng.in_cube(), |rng| rng.in_shell(2.0, 6.0)],
        &mut worst,
    );
}

#[test]
fn p2l_and_l2p_in_f32_match_f64() {
    let mut worst = [
        Worst::new("f32 P2L vs f64, per degree"),
        Worst::new("f32 L2P potential vs f64"),
        Worst::new("f32 L2P gradient vs f64"),
    ];
    compare(
        Kind::Local,
        Basis::Regular,
        [|rng| rng.in_shell(2.0, 6.0), |rng| rng.in_cube()],
        &mut worst,
    );
}

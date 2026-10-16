//! Several charge vectors (Phase 6 T5, C6.3): `p2m_many`, `p2l_many`, `l2p_many` and
//! `m2p_many` against the single-vector operators, vector by vector, bit for bit, in f32
//! and f64, with and without gradients, from nonzero initial values; and their panics.

use nd_fmm_math::RealScalar;
use nd_fmm_ref::{Frame, Workspace, leaf};

use crate::common::{SplitMix64, frames, len, place};

/// The vector counts tested: one (the single-vector operator itself), three and eight.
const VECTORS: [usize; 3] = [1, 3, 8];

/// The degrees tested.
const DEGREES: [usize; 3] = [0, 3, 8];

/// The bits of `values`, for exact comparisons.
fn bits<T: RealScalar>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect()
}

/// `n` values of `T` from `rng`, in [−1, 1].
fn values<T: RealScalar>(rng: &mut SplitMix64, n: usize) -> Vec<T> {
    (0..n).map(|_| T::from_f64(rng.range(-1.0, 1.0))).collect()
}

/// The frame in `T`.
fn frame_in<T: RealScalar>(frame: &Frame<f64>) -> Frame<T> {
    Frame::new(frame.centre.map(T::from_f64), T::from_f64(frame.radius))
}

#[test]
fn several_vectors_are_the_single_vector_operators_bit_for_bit() {
    // Error measure: exact equality of every value, vector by vector.
    expansions_match::<f64>();
    expansions_match::<f32>();
    evaluations_match::<f64>();
    evaluations_match::<f32>();
}

/// P2M and P2L: multipole (local) v of the m-vector operator against the single-vector
/// operator on vector v's charges, both from the same nonzero initial coefficients.
fn expansions_match<T: RealScalar>() {
    let mut rng = SplitMix64::new(0x7105);
    let mut ws = Workspace::<T>::new(8);
    type Single<T> = fn(usize, &Frame<T>, &[[T; 3]], &[T], &mut Workspace<T>, &mut [T]);
    let operators: [(&str, Single<T>, Single<T>, bool); 2] = [
        ("P2M", leaf::p2m, leaf::p2m_many, false),
        ("P2L", leaf::p2l, leaf::p2l_many, true),
    ];
    for (name, single, many, far) in operators {
        for frame in frames().iter().take(3) {
            let sources: Vec<[T; 3]> = (0..11)
                .map(|_| {
                    let u = if far {
                        rng.in_shell(2.0, 6.0)
                    } else {
                        rng.in_cube()
                    };
                    place(frame, u).map(T::from_f64)
                })
                .collect();
            let frame = frame_in::<T>(frame);
            for p in DEGREES {
                for m in VECTORS {
                    let charges: Vec<T> = values(&mut rng, m * sources.len());
                    let start: Vec<T> = values(&mut rng, m * len(p));
                    let mut got = start.clone();
                    many(p, &frame, &sources, &charges, &mut ws, &mut got);
                    for v in 0..m {
                        let q: Vec<T> = charges.iter().skip(v).step_by(m).copied().collect();
                        let mut want = start[v * len(p)..(v + 1) * len(p)].to_vec();
                        single(p, &frame, &sources, &q, &mut ws, &mut want);
                        assert_eq!(
                            bits(&got[v * len(p)..(v + 1) * len(p)]),
                            bits(&want),
                            "{name}, p = {p}, m = {m}, vector {v}"
                        );
                    }
                }
            }
        }
    }
}

/// L2P and M2P: vector v's potentials and gradients of the m-vector operator against the
/// single-vector operator on expansion v, both from the same nonzero initial values.
fn evaluations_match<T: RealScalar>() {
    let mut rng = SplitMix64::new(0x7106);
    let mut ws = Workspace::<T>::new(8);
    type Single<T> =
        fn(usize, &Frame<T>, &[T], &[[T; 3]], &mut Workspace<T>, &mut [T], Option<&mut [[T; 3]]>);
    let operators: [(&str, Single<T>, Single<T>, bool); 2] = [
        ("L2P", leaf::l2p, leaf::l2p_many, false),
        ("M2P", leaf::m2p, leaf::m2p_many, true),
    ];
    for (name, single, many, far) in operators {
        for frame in frames().iter().take(3) {
            let targets: Vec<[T; 3]> = (0..9)
                .map(|_| {
                    let u = if far {
                        rng.in_shell(2.0, 6.0)
                    } else {
                        rng.in_cube()
                    };
                    place(frame, u).map(T::from_f64)
                })
                .collect();
            let frame = frame_in::<T>(frame);
            let n = targets.len();
            for p in DEGREES {
                for m in VECTORS {
                    let coefficients: Vec<T> = values(&mut rng, m * len(p));
                    for gradients in [false, true] {
                        let phi0: Vec<T> = values(&mut rng, m * n);
                        let g0: Vec<[T; 3]> =
                            values(&mut rng, 3 * m * n).as_chunks::<3>().0.to_vec();
                        let (mut phi, mut g) = (phi0.clone(), g0.clone());
                        many(
                            p,
                            &frame,
                            &coefficients,
                            &targets,
                            &mut ws,
                            &mut phi,
                            gradients.then_some(&mut g[..]),
                        );
                        for v in 0..m {
                            let what = format!(
                                "{name}, p = {p}, m = {m}, vector {v}, gradients {gradients}"
                            );
                            let (mut want_phi, mut want_g) = (
                                phi0[v * n..(v + 1) * n].to_vec(),
                                g0[v * n..(v + 1) * n].to_vec(),
                            );
                            single(
                                p,
                                &frame,
                                &coefficients[v * len(p)..(v + 1) * len(p)],
                                &targets,
                                &mut ws,
                                &mut want_phi,
                                gradients.then_some(&mut want_g[..]),
                            );
                            assert_eq!(bits(&phi[v * n..(v + 1) * n]), bits(&want_phi), "{what}");
                            assert_eq!(
                                bits(g[v * n..(v + 1) * n].as_flattened()),
                                bits(want_g.as_flattened()),
                                "{what}: gradient"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "`multipoles` must hold a positive multiple of (p + 1)^2 = 16 values")]
fn p2m_many_rejects_a_partial_multipole() {
    let mut ws = Workspace::new(3);
    let mut multipoles = vec![0.0; 20];
    leaf::p2m_many(
        3,
        &Frame::new([0.0; 3], 1.0),
        &[[0.1; 3]],
        &[1.0],
        &mut ws,
        &mut multipoles,
    );
}

#[test]
#[should_panic(
    expected = "`sources × vectors` and `charges` must have the same length, got 2 and 1"
)]
fn p2l_many_rejects_one_charge_for_two_vectors() {
    let mut ws = Workspace::new(3);
    let mut locals = vec![0.0; 32];
    leaf::p2l_many(
        3,
        &Frame::new([0.0; 3], 1.0),
        &[[3.0; 3]],
        &[1.0],
        &mut ws,
        &mut locals,
    );
}

#[test]
#[should_panic(
    expected = "`targets × vectors` and `potential` must have the same length, got 4 and 2"
)]
fn l2p_many_rejects_one_potential_per_target() {
    let mut ws = Workspace::new(3);
    leaf::l2p_many(
        3,
        &Frame::new([0.0; 3], 1.0),
        &[0.0; 32],
        &[[0.1; 3], [0.2; 3]],
        &mut ws,
        &mut [0.0; 2],
        None,
    );
}

#[test]
#[should_panic(
    expected = "`targets × vectors` and `gradient` must have the same length, got 2 and 1"
)]
fn m2p_many_rejects_a_short_gradient() {
    let mut ws = Workspace::new(3);
    leaf::m2p_many(
        3,
        &Frame::new([0.0; 3], 1.0),
        &[0.0; 16],
        &[[3.0; 3], [4.0; 3]],
        &mut ws,
        &mut [0.0; 2],
        Some(&mut [[0.0; 3]]),
    );
}

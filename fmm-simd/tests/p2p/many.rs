//! Several charge vectors (Phase 6 T5, C6.3): `P2pKernel::evaluate_many` against
//! `P2pKernel::evaluate` vector by vector, on every ISA, in f32 and f64, potential only and
//! with gradients, from nonzero initial outputs.
//!
//! Error measure: exact equality of the bits of every output (as their f64 widening):
//! vector v of `evaluate_many` must be `evaluate` of vector v's charges. The sizes cover
//! every block and tail of the targets (whole blocks, half blocks, single vectors and the
//! padded tail) and of the vector groups (8, 4, 2 and 1), with sources that coincide with
//! targets (r² = 0) and repeated sources.

use nd_fmm_simd::SimdScalar;

use crate::common::kernels;

/// SplitMix64, for deterministic grid-valued inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// k 2⁻¹⁰ for k uniform in [−4096, 4096]: exact in f32 and f64.
    fn grid<T: SimdScalar>(&mut self) -> T {
        let k = (self.next() % 8193) as i32 - 4096;
        T::from_f64(f64::from(k) / 1024.0)
    }

    fn point<T: SimdScalar>(&mut self) -> [T; 3] {
        [self.grid(), self.grid(), self.grid()]
    }
}

/// The bits of `values`, widened to f64.
fn bits<T: SimdScalar>(values: &[T]) -> Vec<u64> {
    values.iter().map(|&v| v.to_f64().to_bits()).collect()
}

#[test]
fn several_vectors_are_evaluate_vector_by_vector_bit_for_bit() {
    several_vectors::<f32>();
    several_vectors::<f64>();
}

fn several_vectors<T: SimdScalar>() {
    let mut rng = Rng(0x7105);
    let kernels = kernels::<T>("several_vectors_are_evaluate_vector_by_vector_bit_for_bit");
    for kernel in kernels {
        for &n_s in &[0usize, 1, 7, 40] {
            for &n_t in &[0usize, 1, 3, 7, 8, 9, 16, 17, 33] {
                for &m in &[1usize, 2, 3, 4, 5, 7, 8, 9, 12, 17] {
                    let mut sources: Vec<[T; 3]> = (0..n_s).map(|_| rng.point()).collect();
                    if n_s > 2 {
                        sources[2] = sources[0];
                    }
                    let charges: Vec<T> = (0..m * n_s).map(|_| rng.grid()).collect();
                    let mut targets: Vec<[T; 3]> = (0..n_t).map(|_| rng.point()).collect();
                    // Coincident pairs: some targets at sources.
                    for (i, target) in targets.iter_mut().enumerate().step_by(3) {
                        if n_s > 0 {
                            *target = sources[i % n_s];
                        }
                    }
                    let potential0: Vec<T> = (0..m * n_t).map(|_| rng.grid()).collect();
                    let gradient0: Vec<[T; 3]> = (0..m * n_t).map(|_| rng.point()).collect();
                    for gradients in [false, true] {
                        let (mut potential, mut gradient) = (potential0.clone(), gradient0.clone());
                        kernel.evaluate_many(
                            &sources,
                            &charges,
                            m,
                            &targets,
                            &mut potential,
                            gradients.then_some(&mut gradient[..]),
                        );
                        for v in 0..m {
                            let what = format!(
                                "{} {}: {n_s} sources, {n_t} targets, m = {m}, vector {v}, \
                                 gradients {gradients}",
                                kernel.isa(),
                                std::any::type_name::<T>()
                            );
                            let q: Vec<T> = charges.iter().skip(v).step_by(m).copied().collect();
                            let range = v * n_t..(v + 1) * n_t;
                            let (mut want_potential, mut want_gradient) = (
                                potential0[range.clone()].to_vec(),
                                gradient0[range.clone()].to_vec(),
                            );
                            kernel.evaluate(
                                &sources,
                                &q,
                                &targets,
                                &mut want_potential,
                                gradients.then_some(&mut want_gradient[..]),
                            );
                            assert_eq!(
                                bits(&potential[range.clone()]),
                                bits(&want_potential),
                                "{what}: potential"
                            );
                            assert_eq!(
                                bits(gradient[range].as_flattened()),
                                bits(want_gradient.as_flattened()),
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
#[should_panic(expected = "`vectors` must be at least 1")]
fn evaluate_many_rejects_no_vector() {
    let kernel = nd_fmm_simd::P2pKernel::<f64>::detect();
    kernel.evaluate_many(&[], &[], 0, &[], &mut [], None);
}

#[test]
#[should_panic(
    expected = "`sources × vectors` and `charges` must have the same length, got 4 and 3"
)]
fn evaluate_many_rejects_short_charges() {
    let kernel = nd_fmm_simd::P2pKernel::<f64>::detect();
    kernel.evaluate_many(&[[0.0; 3]; 2], &[1.0; 3], 2, &[], &mut [], None);
}

#[test]
#[should_panic(
    expected = "`targets × vectors` and `potential` must have the same length, got 6 and 3"
)]
fn evaluate_many_rejects_one_potential_per_target() {
    let kernel = nd_fmm_simd::P2pKernel::<f32>::detect();
    kernel.evaluate_many(&[], &[], 2, &[[0.0; 3]; 3], &mut [0.0; 3], None);
}

#[test]
#[should_panic(
    expected = "`targets × vectors` and `gradient` must have the same length, got 6 and 3"
)]
fn evaluate_many_rejects_a_short_gradient() {
    let kernel = nd_fmm_simd::P2pKernel::<f32>::detect();
    kernel.evaluate_many(
        &[],
        &[],
        2,
        &[[0.0; 3]; 3],
        &mut [0.0; 6],
        Some(&mut [[0.0; 3]; 3]),
    );
}

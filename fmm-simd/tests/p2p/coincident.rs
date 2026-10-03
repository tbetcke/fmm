//! Coincident points: targets equal to the sources (the same slice), duplicated
//! sources, a mapped neighbour source that rounds onto a target, and signed zeros. The
//! pairs with r² = 0 contribute nothing: the results are finite and within the sum
//! tolerance of `direct_sum`, which skips the same pairs (on these inputs r² = 0
//! exactly when the points coincide). The scalar path equals the reference bit for
//! bit.

use nd_fmm_simd::SimdScalar;

use crate::common::{
    Oracle, Outputs, Precision, Problem, Rng, Worst, assert_within, bits, kernels, run,
    run_reference, seeded,
};

/// Checks the outputs of `evaluate` on `problem`, for each available ISA, potential
/// only and with gradient: finite and within the sum tolerance of `direct_sum`, and, on
/// the scalar path, the reference's bits. `evaluate` runs one kernel from the initial
/// outputs of `problem`.
fn check<T: Precision>(
    test: &str,
    problem: &Problem<T>,
    evaluate: &dyn Fn(nd_fmm_simd::P2pKernel<T>, bool) -> Outputs<T>,
) {
    let oracle = Oracle::new(problem);
    for with_gradient in [false, true] {
        let mut worst = Worst::new(format!(
            "{test} ({}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for kernel in kernels::<T>(test) {
            let outputs = evaluate(kernel, with_gradient);
            let errors = oracle.errors(&outputs);
            assert_within(errors, oracle.tolerance::<T>(), &kernel.isa());
            worst.record_sum::<T>(kernel.isa(), errors, &oracle, with_gradient);
            if kernel.isa() == nd_fmm_simd::Isa::Scalar {
                assert_eq!(
                    bits(&outputs),
                    bits(&run_reference(problem, with_gradient)),
                    "scalar against the reference"
                );
            }
        }
        worst.print(1.0, "(relative to the term magnitudes)");
    }
}

/// Targets equal to the sources, passed as the same slice: every self-interaction
/// pair is skipped.
fn targets_are_sources<T: Precision>(test: &str) {
    for n in [1, 2, 5, 16, 33] {
        let mut raw = seeded(n, n, 7 + n as u64);
        raw.targets = raw.sources.clone();
        let problem = raw.cast::<T>();
        check(test, &problem, &|kernel, with_gradient| {
            let (mut potential, mut gradient) = crate::common::initial(&problem, with_gradient);
            kernel.evaluate(
                &problem.sources,
                &problem.charges,
                &problem.sources,
                &mut potential,
                gradient.as_deref_mut(),
            );
            (potential, gradient)
        });
    }
}

#[test]
fn targets_equal_to_sources_f32() {
    targets_are_sources::<f32>("targets_equal_to_sources_f32");
}

#[test]
fn targets_equal_to_sources_f64() {
    targets_are_sources::<f64>("targets_equal_to_sources_f64");
}

/// Each source three times, and targets at the sources and elsewhere: every copy of a
/// source that coincides with a target is skipped.
fn duplicated_sources<T: Precision>(test: &str) {
    let mut raw = seeded(6, 4, 11);
    raw.sources = raw.sources.repeat(3);
    raw.charges = raw.charges.repeat(3);
    raw.targets.extend_from_slice(&raw.sources[..5]);
    raw.potential.extend_from_slice(&[3, -2, 1, 0, 5]);
    raw.gradient.extend_from_slice(&[[1, 2, 3]; 5]);
    let problem = raw.cast::<T>();
    check(test, &problem, &|kernel, with_gradient| {
        run(kernel, &problem, with_gradient)
    });
}

#[test]
fn duplicated_sources_f32() {
    duplicated_sources::<f32>("duplicated_sources_f32");
}

#[test]
fn duplicated_sources_f64() {
    duplicated_sources::<f64>("duplicated_sources_f64");
}

/// A neighbour source that `LaplaceOperator`'s mapping rounds onto a target
/// (CONVENTIONS §3.13, "Coincident pairs"): the target at u_t = (1, a, b) on the face
/// of its leaf, and a source of the same-level neighbour at ĉ = (2, 0, 0), r̂ = 1, with
/// u_s = (−(1 − u_T), a, b), a point distinct from the target. Then
/// ŷ = fl(2 − (1 − u_T)) = fl(1 + u_T) = 1 (a tie, to even), so ŷ equals u_t and
/// the pair is skipped. Other sources and targets are random.
fn mapped_neighbour<T: Precision>(test: &str) {
    let mut rng = Rng(0x0a9e_d001);
    let n = 11;
    let mut targets: Vec<[T; 3]> = (0..n).map(|_| rng.point()).collect();
    let (a, b) = (targets[3][1], targets[3][2]);
    targets[3][0] = T::one();
    let mut u_s: Vec<[T; 3]> = (0..n).map(|_| rng.point()).collect();
    let u = T::from_f64(T::U);
    u_s[6] = [-(T::one() - u), a, b];
    let two = T::from_f64(2.0);
    let sources: Vec<[T; 3]> = u_s.iter().map(|p| [two + p[0], p[1], p[2]]).collect();
    assert!(u_s[6][0] != -T::one() && sources[6] == targets[3]);
    let problem = Problem {
        charges: (0..n)
            .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
            .collect(),
        potential: (0..n)
            .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
            .collect(),
        gradient: (0..n).map(|_| rng.point()).collect(),
        sources,
        targets,
    };
    check(test, &problem, &|kernel, with_gradient| {
        run(kernel, &problem, with_gradient)
    });
}

#[test]
fn mapped_neighbour_rounds_onto_target_f32() {
    mapped_neighbour::<f32>("mapped_neighbour_rounds_onto_target_f32");
}

#[test]
fn mapped_neighbour_rounds_onto_target_f64() {
    mapped_neighbour::<f64>("mapped_neighbour_rounds_onto_target_f64");
}

/// +0 and −0 compare equal and their difference is ±0, so (0, 0, 0) and (−0, 0, −0)
/// coincide under both rules; the other term is exact, so every ISA gives the same
/// bits.
fn signed_zeros<T: SimdScalar>(test: &str) {
    let zero = T::zero();
    let sources = [[-zero, zero, -zero], [T::one(), zero, zero]];
    let charges = [T::one(), T::from_f64(2.0)];
    let targets = [[zero, zero, zero]];
    for kernel in kernels::<T>(test) {
        let mut potential = [zero];
        let mut gradient = [[zero; 3]];
        kernel.evaluate(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let outputs = (potential.to_vec(), Some(gradient.to_vec()));
        let two = T::from_f64(2.0);
        let expected = (vec![two], Some(vec![[two, zero, zero]]));
        assert_eq!(bits(&outputs), bits(&expected), "{}", kernel.isa());
    }
}

#[test]
fn signed_zeros_coincide_f32() {
    signed_zeros::<f32>("signed_zeros_coincide_f32");
}

#[test]
fn signed_zeros_coincide_f64() {
    signed_zeros::<f64>("signed_zeros_coincide_f64");
}

//! Empty inputs and accumulation onto nonzero outputs.

use nd_fmm_simd::SimdScalar;

use crate::common::{
    Oracle, Precision, Problem, Rng, Worst, assert_within, bits, initial, kernels, run,
    run_reference, scalar_kernel, seeded, uniform,
};

/// No sources: the outputs keep their bits. No targets: nothing to write.
fn empty_inputs<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (n_sources, n_targets) in [(0, 0), (0, 5), (0, 9), (5, 0)] {
            let problem = seeded(n_sources, n_targets, 3).cast::<T>();
            for with_gradient in [false, true] {
                let outputs = run(kernel, &problem, with_gradient);
                assert_eq!(
                    bits(&outputs),
                    bits(&initial(&problem, with_gradient)),
                    "{} n_s = {n_sources}, n_t = {n_targets}",
                    kernel.isa()
                );
            }
        }
    }
}

#[test]
fn empty_inputs_f32() {
    empty_inputs::<f32>("empty_inputs_f32");
}

#[test]
fn empty_inputs_f64() {
    empty_inputs::<f64>("empty_inputs_f64");
}

/// Accumulation onto nonzero initial outputs:
/// - within the sum tolerance of `direct_sum` from the same initial values (the
///   error relative to the initial value plus the term magnitudes);
/// - a second evaluation adds the same terms again onto the first result, in the same
///   order, so it equals one evaluation of the sources listed twice, bit for bit;
/// - on the scalar path, both as the reference, bit for bit.
fn accumulates<T: Precision>(test: &str) {
    let mut problem: Problem<T> = uniform(57, 19, 5);
    let mut rng = Rng(0xacc0);
    for (v, g) in problem.potential.iter_mut().zip(&mut problem.gradient) {
        *v = T::from_f64(rng.uniform(-50.0, 50.0));
        *g = [0; 3].map(|_| T::from_f64(rng.uniform(-50.0, 50.0)));
    }
    let oracle = Oracle::new(&problem);
    let twice = Problem {
        sources: problem.sources.repeat(2),
        charges: problem.charges.repeat(2),
        ..problem.clone()
    };
    for with_gradient in [false, true] {
        let mut worst = Worst::new(format!(
            "{test} ({}, gradient: {with_gradient})",
            std::any::type_name::<T>()
        ));
        for kernel in kernels::<T>(test) {
            let once = run(kernel, &problem, with_gradient);
            assert_ne!(
                bits(&once),
                bits(&initial(&problem, with_gradient)),
                "the problem has nonzero terms"
            );
            let errors = oracle.errors(&once);
            assert_within(errors, oracle.tolerance::<T>(), &kernel.isa());
            worst.record_sum::<T>(kernel.isa(), errors, &oracle, with_gradient);

            let (mut potential, mut gradient) = once.clone();
            kernel.evaluate(
                &problem.sources,
                &problem.charges,
                &problem.targets,
                &mut potential,
                gradient.as_deref_mut(),
            );
            assert_eq!(
                bits(&(potential, gradient)),
                bits(&run(kernel, &twice, with_gradient)),
                "{} gradient: {with_gradient}",
                kernel.isa()
            );
        }
        worst.print(
            1.0,
            "(relative to the initial value and the term magnitudes)",
        );
    }
    let scalar = scalar_kernel::<T>(test);
    for with_gradient in [false, true] {
        assert_eq!(
            bits(&run(scalar, &twice, with_gradient)),
            bits(&run_reference(&twice, with_gradient))
        );
    }
}

#[test]
fn accumulates_onto_nonzero_output_f32() {
    accumulates::<f32>("accumulates_onto_nonzero_output_f32");
}

#[test]
fn accumulates_onto_nonzero_output_f64() {
    accumulates::<f64>("accumulates_onto_nonzero_output_f64");
}

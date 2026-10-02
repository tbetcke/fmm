//! Empty inputs and accumulation onto nonzero outputs.

use nd_fmm_simd::SimdScalar;

use crate::common::{bits, initial, kernels, run, run_reference, seeded};

/// No sources: the outputs keep their bits. No targets: nothing to write.
fn empty_inputs<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        for (n_sources, n_targets) in [(0, 0), (0, 5), (5, 0)] {
            let problem = seeded(n_sources, n_targets, 3).cast::<T>();
            for with_gradient in [false, true] {
                let outputs = run(kernel, &problem, with_gradient);
                assert_eq!(
                    bits(&outputs),
                    bits(&initial(&problem, with_gradient)),
                    "{} n_s = {n_sources}, n_t = {n_targets}",
                    kernel.isa()
                );
                assert_eq!(
                    bits(&outputs),
                    bits(&run_reference(&problem, with_gradient))
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

/// Accumulation: from a nonzero output, as the reference, and a second evaluation adds
/// the same terms again onto the first result, in the same order.
fn accumulates<T: SimdScalar>(test: &str) {
    let problem = seeded(23, 9, 5).cast::<T>();
    for kernel in kernels::<T>(test) {
        for with_gradient in [false, true] {
            let once = run(kernel, &problem, with_gradient);
            assert_ne!(
                bits(&once),
                bits(&initial(&problem, with_gradient)),
                "the problem has nonzero terms"
            );
            assert_eq!(bits(&once), bits(&run_reference(&problem, with_gradient)));

            let (mut potential, mut gradient) = once.clone();
            kernel.evaluate(
                &problem.sources,
                &problem.charges,
                &problem.targets,
                &mut potential,
                gradient.as_deref_mut(),
            );
            let mut again = problem.clone();
            again.potential = once.0.clone();
            if let Some(g) = &once.1 {
                again.gradient = g.clone();
            }
            assert_eq!(
                bits(&(potential, gradient)),
                bits(&run_reference(&again, with_gradient)),
                "{} gradient: {with_gradient}",
                kernel.isa()
            );
        }
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

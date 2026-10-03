//! Smoke run of the core of the `p2p_kernels` example: one small W1 cell (n_t = 8) per
//! available ISA and the reference, in f32 and f64, with and without gradients, in both
//! forms, checked against `direct_sum`, with a short timing. Prints the ISAs it ran.
//!
//! No MPI: the benchmark core never initialises it. Error measure: the largest error
//! relative to the term magnitudes ([`Accuracy::max`]), against requirement 2 of
//! docs/design/simd-p2p.md §3 ([`passes`]).

use nd_fmm_simd::{Isa, SimdScalar};
use nd_fmm_validate::p2p_kernels::{
    Accuracy, Cell, Form, Kernel, Outputs, check, oracles, pairs_per_second, passes,
    per_pair_equals_gathered, precision_name, sum_tolerance,
};

fn smoke<T: SimdScalar>() {
    let kernels = Kernel::<T>::all();
    let ran: Vec<String> = kernels.iter().map(Kernel::name).collect();
    println!(
        "{}: kernels {ran:?} (available ISAs: {:?})",
        precision_name::<T>(),
        Isa::available().collect::<Vec<_>>()
    );
    assert_eq!(kernels.len(), 1 + Isa::available().count());
    let pool = Cell::W1 {
        targets: 8,
        form: Form::Gathered,
    }
    .pool::<T>();
    let oracles = oracles(&pool);
    for gradients in [false, true] {
        let reference = check(
            &Kernel::Reference,
            &pool,
            &oracles,
            Form::Gathered,
            gradients,
        );
        assert!(
            reference.max() <= sum_tolerance::<T>(),
            "the reference: {reference:?}"
        );
        for kernel in &kernels {
            for form in [Form::Gathered, Form::PerPair] {
                let accuracy = check(kernel, &pool, &oracles, form, gradients);
                println!(
                    "{} {} {} gradients={gradients}: {accuracy:?}",
                    precision_name::<T>(),
                    kernel.name(),
                    form.name()
                );
                assert!(
                    passes::<T>(accuracy, reference),
                    "{} {form:?}: {accuracy:?}",
                    kernel.name()
                );
                assert_eq!(accuracy.l2_gradient > 0.0, gradients);
            }
            if kernel.isa().is_some() {
                assert!(per_pair_equals_gathered(kernel, &pool[0], gradients));
            }
        }
    }
    // A failed check: an error far above the tolerance.
    assert!(!passes::<T>(
        Accuracy {
            max_potential: 1.0,
            ..Accuracy::default()
        },
        Accuracy::default()
    ));
    // The timing loop runs (its value is not checked).
    let mut outputs = Outputs::for_pool(&pool[..2]);
    let rate = pairs_per_second(pool[0].pairs(), 2, |i| {
        Kernel::<T>::Simd(nd_fmm_simd::P2pKernel::detect()).evaluate(
            &pool[i],
            Form::Gathered,
            true,
            &mut outputs[i],
        );
    });
    assert!(rate > 0.0);
}

#[test]
fn p2p_kernels_core_f32() {
    smoke::<f32>();
}

#[test]
fn p2p_kernels_core_f64() {
    smoke::<f64>();
}

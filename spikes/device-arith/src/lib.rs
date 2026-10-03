//! Phase 4 T3 spike: device arithmetic per CubeCL backend (docs/phase4/T3-device-arithmetic.md).
//! See REPORT.md.

pub mod backend;
pub mod compiler;
pub mod cpu_p2p;
pub mod domain;
pub mod geometry;
pub mod leafops;
pub mod p2p;
pub mod pairs;
pub mod primitives;
pub mod real;

/// The smoke test of the brief: the primitive checks, the r² = 0 check and one W1 cell of
/// the CPU-shaped P2P per precision, on the CPU runtime at a reduced size.
#[cfg(all(test, feature = "cpu"))]
mod smoke {
    use crate::backend::Backend;
    use crate::real::Real;
    use crate::{cpu_p2p, domain, p2p, pairs, primitives};
    use nd_fmm_validate::p2p_kernels::{Form, Kernel, W1_SEED, check, oracles, passes, w1};

    /// The CPU runtime's `sqrt` and divisions are correctly rounded and its inverse square
    /// roots within 1.5 u_T (fl(1/fl(√x)); CubeCL polyfills `inverse_sqrt` so on LLVM).
    /// Error measure: the largest error in u_T against the exact value, and bit equality
    /// with the correctly rounded host result.
    fn primitives_on<T: Real>(inputs: &primitives::Inputs<T>) {
        let client = Backend::Cpu.client();
        for op in primitives::Op::ALL {
            let a = primitives::measure(&client, Backend::Cpu, op, inputs);
            println!(
                "cpu {} {}: {:.3} u_T, {} differ",
                T::NAME,
                op.name(),
                a.max_error,
                a.differ
            );
            let bound = match op {
                primitives::Op::Sqrt
                | primitives::Op::Div
                | primitives::Op::OneDiv
                | primitives::Op::Recip => 1.0,
                _ => 1.5,
            };
            assert!(
                a.max_error <= bound,
                "{} {}: {}",
                T::NAME,
                op.name(),
                a.max_error
            );
            if bound == 1.0 {
                assert_eq!(
                    a.differ,
                    0,
                    "{} {} is not correctly rounded",
                    T::NAME,
                    op.name()
                );
            }
        }
    }

    #[test]
    fn primitives_cpu() {
        println!("backends exercised: cpu");
        primitives_on::<f32>(&primitives::powers_of_two(5));
        primitives_on::<f64>(&primitives::log_uniform(20_000, 7));
    }

    /// r² = 0 exactly for coincident stored points and every other r² in [2⁻¹⁰⁶, 2⁷], in
    /// every formulation, on adversarial pairs of a few levels and on random pairs.
    /// Error measure: exact (counts of violations must be zero).
    fn domain_on<T: Real>() {
        let client = Backend::Cpu.client();
        for list in [
            pairs::adversarial::<T>(1, &[0, 1, 9, 16]),
            pairs::random::<T>(2, 50),
        ] {
            let o = domain::check(&list, &domain::run(&client, Backend::Cpu, &list));
            assert!(o.coincident > 0, "no coincident pair generated");
            for f in 0..domain::FORMULATIONS.len() {
                assert!(o.holds(f), "{} {}: {o:?}", T::NAME, domain::FORMULATIONS[f]);
            }
        }
    }

    #[test]
    fn r2_rule_cpu() {
        println!("backends exercised: cpu");
        domain_on::<f32>();
        domain_on::<f64>();
    }

    /// The CPU-shaped P2P against `direct_sum` on one W1 cell (n_t = 24) per precision,
    /// with and without gradients, at 1 and 12 units, and its pair terms within the C3S.4
    /// contract (8 u_T for φ, 16 u_T for ∇φ). Error measure: `p2p_kernels::passes` (1e-6 /
    /// 1e-14 relative to the term magnitudes, or twice the reference's error), and u_T.
    fn cpu_p2p_on<T: Real>() {
        let client = Backend::Cpu.client();
        let pool = w1::<T>(24, W1_SEED);
        let oracles = oracles(&pool);
        let checked = &pool[..oracles.len()];
        for gradients in [false, true] {
            let reference = check(
                &Kernel::<T>::Reference,
                checked,
                &oracles,
                Form::Gathered,
                gradients,
            );
            for units in [1, 12] {
                let a = cpu_p2p::accuracy_on(&client, checked, &oracles, gradients, units);
                assert!(
                    passes::<T>(a, reference),
                    "{} gradients {gradients}: {a:?}",
                    T::NAME
                );
            }
        }
        let (n, k) = cpu_p2p::layout::<T>();
        let e = cpu_p2p::pair_errors(&client, &p2p::domain_pairs::<T>(4096, n * k, 9));
        assert!(
            e.potential <= 8.0 && e.gradient <= 16.0,
            "{} {e:?}",
            T::NAME
        );
    }

    #[test]
    fn cpu_shaped_p2p_cpu() {
        println!("backends exercised: cpu");
        cpu_p2p_on::<f32>();
        cpu_p2p_on::<f64>();
    }
}

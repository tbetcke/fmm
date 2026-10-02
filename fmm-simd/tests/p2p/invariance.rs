//! Chunk invariance, target-position invariance and determinism, bit for bit
//! (docs/design/simd-p2p.md §3, requirements 4 and 5), on random problems with
//! coincident pairs.

use nd_fmm_simd::{P2pKernel, SimdScalar};
use proptest::prelude::*;

use crate::common::{Problem, bits, kernels, problem, run, runner};

/// Evaluating sources `[..k]` and then `[k..]`, for every k, equals evaluating all of
/// them at once.
fn chunk_invariance<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        runner()
            .run(&problem(24, 11), |raw| {
                let problem = raw.cast::<T>();
                for with_gradient in [false, true] {
                    let whole = bits(&run(kernel, &problem, with_gradient));
                    for k in 0..=problem.sources.len() {
                        let first = Problem {
                            sources: problem.sources[..k].to_vec(),
                            charges: problem.charges[..k].to_vec(),
                            ..problem.clone()
                        };
                        let (potential, gradient) = run(kernel, &first, with_gradient);
                        let second = Problem {
                            sources: problem.sources[k..].to_vec(),
                            charges: problem.charges[k..].to_vec(),
                            potential,
                            gradient: gradient.unwrap_or_else(|| problem.gradient.clone()),
                            ..problem.clone()
                        };
                        let split = bits(&run(kernel, &second, with_gradient));
                        prop_assert_eq!(&split, &whole, "{} k = {}", kernel.isa(), k);
                    }
                }
                Ok(())
            })
            .unwrap();
    }
}

#[test]
fn chunk_invariance_f32() {
    chunk_invariance::<f32>("chunk_invariance_f32");
}

#[test]
fn chunk_invariance_f64() {
    chunk_invariance::<f64>("chunk_invariance_f64");
}

/// The single-target problem of target `i`.
fn single<T: SimdScalar>(problem: &Problem<T>, i: usize) -> Problem<T> {
    Problem {
        targets: vec![problem.targets[i]],
        potential: vec![problem.potential[i]],
        gradient: vec![problem.gradient[i]],
        ..problem.clone()
    }
}

/// The problem with the targets in the order `order`.
fn permuted<T: SimdScalar>(problem: &Problem<T>, order: &[usize]) -> Problem<T> {
    Problem {
        targets: order.iter().map(|&i| problem.targets[i]).collect(),
        potential: order.iter().map(|&i| problem.potential[i]).collect(),
        gradient: order.iter().map(|&i| problem.gradient[i]).collect(),
        ..problem.clone()
    }
}

/// A target's result is the same alone, at any position, among any number of targets,
/// and from run to run.
fn target_position_invariance<T: SimdScalar>(test: &str) {
    for kernel in kernels::<T>(test) {
        runner()
            .run(&(problem(24, 17), any::<usize>()), |(raw, rotation)| {
                let problem = raw.cast::<T>();
                let n = problem.targets.len();
                for with_gradient in [false, true] {
                    let whole = per_target(kernel, &problem, with_gradient);
                    prop_assert_eq!(
                        &per_target(kernel, &problem, with_gradient),
                        &whole,
                        "{} is not deterministic",
                        kernel.isa()
                    );
                    for (i, expected) in whole.iter().enumerate() {
                        let alone = per_target(kernel, &single(&problem, i), with_gradient);
                        prop_assert_eq!(&alone[0], expected, "{} target {}", kernel.isa(), i);
                    }
                    // Reversed, rotated, and every prefix (each target count).
                    let reversed: Vec<usize> = (0..n).rev().collect();
                    let shift = if n == 0 { 0 } else { rotation % n };
                    let rotated: Vec<usize> = (0..n).map(|i| (i + shift) % n).collect();
                    let prefixes = (0..=n).map(|m| (0..m).collect::<Vec<usize>>());
                    for order in [reversed, rotated].into_iter().chain(prefixes) {
                        let moved = per_target(kernel, &permuted(&problem, &order), with_gradient);
                        for (result, &i) in moved.iter().zip(&order) {
                            prop_assert_eq!(result, &whole[i], "{} target {}", kernel.isa(), i);
                        }
                    }
                }
                Ok(())
            })
            .unwrap();
    }
}

/// The output bits of each target of `problem`.
fn per_target<T: SimdScalar>(
    kernel: P2pKernel<T>,
    problem: &Problem<T>,
    with_gradient: bool,
) -> Vec<Vec<u64>> {
    let (potential, gradient) = run(kernel, problem, with_gradient);
    (0..potential.len())
        .map(|i| {
            let outputs = (vec![potential[i]], gradient.as_ref().map(|g| vec![g[i]]));
            bits(&outputs)
        })
        .collect()
}

#[test]
fn target_position_invariance_f32() {
    target_position_invariance::<f32>("target_position_invariance_f32");
}

#[test]
fn target_position_invariance_f64() {
    target_position_invariance::<f64>("target_position_invariance_f64");
}

//! Chunk invariance, target-position invariance and determinism, bit for bit
//! (docs/design/simd-p2p.md §3, requirements 4 and 5), on random problems with
//! coincident pairs and on large seeded sets.

use nd_fmm_simd::{P2pKernel, SimdScalar};
use proptest::prelude::*;

use crate::common::{BLOCK, Problem, Rng, bits, kernels, problem, run, runner, uniform};

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
                        let split = split(kernel, &problem, k, with_gradient);
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

/// The problem evaluated as sources `[..k]`, then `[k..]`.
fn split<T: SimdScalar>(
    kernel: P2pKernel<T>,
    problem: &Problem<T>,
    k: usize,
    with_gradient: bool,
) -> Vec<u64> {
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
    bits(&run(kernel, &second, with_gradient))
}

/// Large seeded sets, split at random k: 1,000 and 4,096 sources, 3 · BLOCK + 1
/// targets.
fn chunk_invariance_large<T: SimdScalar>(test: &str) {
    let mut rng = Rng(0xc4a2_0001);
    for kernel in kernels::<T>(test) {
        for n_sources in [1000, 4096] {
            let problem: Problem<T> = uniform(n_sources, 3 * BLOCK + 1, n_sources as u64);
            for with_gradient in [false, true] {
                let whole = bits(&run(kernel, &problem, with_gradient));
                for _ in 0..8 {
                    let k = rng.below(n_sources + 1);
                    assert_eq!(
                        split(kernel, &problem, k, with_gradient),
                        whole,
                        "{} n_s = {n_sources}, k = {k}",
                        kernel.isa()
                    );
                }
            }
        }
    }
}

#[test]
fn chunk_invariance_large_f32() {
    chunk_invariance_large::<f32>("chunk_invariance_large_f32");
}

#[test]
fn chunk_invariance_large_f64() {
    chunk_invariance_large::<f64>("chunk_invariance_large_f64");
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
                    // Shifted: after 1 to BLOCK other targets, and before as many, so
                    // that each target lands in every lane and block position.
                    for shift in 1..=BLOCK {
                        let mut order: Vec<usize> = (0..shift).map(|j| j % n.max(1)).collect();
                        if n == 0 {
                            break;
                        }
                        let head = order.len();
                        order.extend(0..n);
                        order.extend((0..shift).map(|j| (j + 1) % n));
                        let moved = per_target(kernel, &permuted(&problem, &order), with_gradient);
                        for (i, expected) in whole.iter().enumerate() {
                            prop_assert_eq!(
                                &moved[head + i],
                                expected,
                                "{} target {} shifted by {}",
                                kernel.isa(),
                                i,
                                shift
                            );
                        }
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

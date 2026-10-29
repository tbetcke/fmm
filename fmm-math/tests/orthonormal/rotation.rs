//! The rotation blocks: orthogonality, the Laplace blocks conjugated by N, the action
//! on the harmonics, and the π − β rule (tests 7, 8, 9 and 10 of the plan).

use nd_fmm_math::Layout;
use nd_fmm_math::orthonormal::harmonics;
use nd_fmm_math::orthonormal::rotation::{
    Scratch, blocks, compact_blocks_len, compact_offset, supplement, y_blocks,
};
use nd_fmm_math::rotation::{block_range, blocks_len, euler_zyz};
use std::f64::consts::PI;

use crate::common::SplitMix64;
use crate::support::{max_abs_diff, orthonormal_block, square_mul};

/// The octree's polar angles and two generic ones, as the fixture's.
pub fn angles() -> Vec<f64> {
    let m2m = (1.0f64 / 3.0f64.sqrt()).acos();
    vec![m2m, PI - m2m, (1.0f64 / 3.0).atan(), PI / 2.0, 1.234, 1e-3]
}

/// Compact y-rotation blocks to degree p for the angle β.
pub fn y(p: usize, beta: f64) -> Vec<f64> {
    let (s, x) = beta.sin_cos();
    let mut out = vec![f64::NAN; compact_blocks_len(p)];
    y_blocks(p, x, s, &mut out, &mut Scratch::new(p));
    out
}

/// The full (2n + 1)² block of degree n from compact blocks, slots −n..n, row-major.
pub fn full(n: usize, compact: &[f64]) -> Vec<f64> {
    let w = 2 * n + 1;
    let block = &compact[compact_offset(n)..compact_offset(n + 1)];
    let (cos, sin) = block.split_at((n + 1) * (n + 1));
    let mut out = vec![0.0; w * w];
    for a in 0..=n {
        for b in 0..=n {
            out[(a + n) * w + (b + n)] = cos[a * (n + 1) + b];
        }
    }
    for a in 1..=n {
        for b in 1..=n {
            out[(n - a) * w + (n - b)] = sin[(a - 1) * n + (b - 1)];
        }
    }
    out
}

/// ‖O Oᵀ − I‖_max of a row-major block of width w.
fn orthogonality(w: usize, o: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..w {
        for j in 0..w {
            let dot: f64 = (0..w).map(|k| o[i * w + k] * o[j * w + k]).sum();
            let identity = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - identity).abs());
        }
    }
    worst
}

/// Test 7: orthogonal to 1e-13, n ≤ 100, at the six angles.
#[test]
fn y_blocks_are_orthogonal() {
    let p = 100;
    for beta in angles() {
        let compact = y(p, beta);
        let worst = (0..=p).fold(0.0f64, |m, n| {
            m.max(orthogonality(2 * n + 1, &full(n, &compact)))
        });
        assert!(worst <= 1e-13, "β = {beta}: {worst:e}");
    }
}

/// Test 7 at large degree: n = 500, 700 and 1000.
#[test]
#[ignore = "release mode: block sets to degree 1000"]
fn y_blocks_are_orthogonal_at_large_degree() {
    let p = 1000;
    for beta in angles() {
        let compact = y(p, beta);
        for n in [500, 700, 1000] {
            let worst = orthogonality(2 * n + 1, &full(n, &compact));
            println!("β = {beta:.6} n = {n:4} ‖O Oᵀ − I‖ = {worst:.2e}");
            assert!(worst <= 1e-13, "β = {beta}, n = {n}: {worst:e}");
        }
    }
}

/// Test 8: `blocks(α, β, γ)` equals N Dⁿ N⁻¹ of the Laplace `rotation::blocks` at the same
/// rotation, p ≤ 30.
#[test]
fn equal_the_laplace_blocks_conjugated_by_n() {
    let p = 30;
    let mut rng = SplitMix64::new(5);
    let mut scratch = Scratch::new(p);
    let mut worst = 0.0f64;
    for _ in 0..20 {
        let (alpha, beta, gamma) = (rng.range(-PI, PI), rng.range(0.0, PI), rng.range(-PI, PI));
        let mut laplace = vec![0.0; blocks_len(p)];
        nd_fmm_math::rotation::blocks(p, &euler_zyz(alpha, beta, gamma), &mut laplace);
        let mut ours = vec![0.0; blocks_len(p)];
        blocks(p, alpha, beta, gamma, &mut ours, &mut scratch);
        for n in 0..=p {
            worst = worst.max(max_abs_diff(
                &ours[block_range(n)],
                &orthonormal_block(&laplace, n),
            ));
        }
    }
    assert!(worst <= 1e-12, "worst {worst:e}");
}

/// Test 8: Yₙ(Q x̂) = Oⁿ(Q) Yₙ(x̂) at p = 50, relative to the degree's largest value; and
/// the blocks of a product are the product of the blocks.
#[test]
fn rotate_the_harmonics() {
    let p = 50;
    let layout = Layout::new(p);
    let mut rng = SplitMix64::new(9);
    let mut scratch = Scratch::new(p);
    let mut worst = 0.0f64;
    for _ in 0..10 {
        let (alpha, beta, gamma) = (rng.range(-PI, PI), rng.range(0.0, PI), rng.range(-PI, PI));
        let q = euler_zyz(alpha, beta, gamma);
        let mut o = vec![0.0; blocks_len(p)];
        blocks(p, alpha, beta, gamma, &mut o, &mut scratch);
        for _ in 0..5 {
            let x = rng.unit_vector();
            let qx = [0, 1, 2].map(|i| (0..3).map(|j| q[i][j] * x[j]).sum::<f64>());
            let (mut y, mut yq) = (vec![0.0; layout.len()], vec![0.0; layout.len()]);
            harmonics(p, x, &mut y);
            harmonics(p, qx, &mut yq);
            for (n, range) in layout.degrees() {
                let rows = o[block_range(n)].chunks_exact(2 * n + 1);
                let scale = yq[range.clone()].iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for (row, expected) in rows.zip(&yq[range.clone()]) {
                    let value: f64 = row.iter().zip(&y[range.clone()]).map(|(a, b)| a * b).sum();
                    worst = worst.max((value - expected).abs() / scale);
                }
            }
        }
    }
    assert!(worst <= 1e-12, "worst {worst:e}");
    // O(R_y(β₁)) O(R_y(β₂)) = O(R_y(β₁ + β₂)).
    let (b1, b2) = (0.4, 0.9);
    let (o1, o2, o12) = (y(p, b1), y(p, b2), y(p, b1 + b2));
    for n in [0, 1, 7, 50] {
        let w = 2 * n + 1;
        let product = square_mul(w, &full(n, &o1), &full(n, &o2));
        assert!(max_abs_diff(&product, &full(n, &o12)) <= 1e-13, "n = {n}");
    }
}

/// Test 9: `supplement` turns the blocks of β into those of π − β, p = 100.
#[test]
fn supplement_gives_pi_minus_beta() {
    let p = 100;
    for beta in angles() {
        let mut signed = y(p, beta);
        supplement(p, &mut signed);
        let direct = y(p, PI - beta);
        let worst = max_abs_diff(&signed, &direct);
        assert!(worst <= 1e-13, "β = {beta}: {worst:e}");
    }
}

/// Test 10: lower degrees do not depend on p (bit for bit); β = 0 gives the identity to
/// round-off (the recursion rounds, so not exactly).
#[test]
fn are_consistent_across_degrees() {
    let (small, large) = (y(10, 0.7), y(40, 0.7));
    assert_eq!(small, large[..small.len()]);
    let identity = y(20, 0.0);
    for n in 0..=20 {
        let w = 2 * n + 1;
        let block = full(n, &identity);
        for i in 0..w {
            for j in 0..w {
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (block[i * w + j] - expected).abs();
                assert!(error <= 1e-15, "n {n} ({i}, {j}): {error:e}");
            }
        }
    }
}

/// Length checks.
#[test]
#[should_panic(expected = "must have length")]
fn wrong_length_panics() {
    let mut out = vec![0.0; 3];
    y_blocks(1, 1.0, 0.0, &mut out, &mut Scratch::new(1));
}

/// Test 10: f32 against f64 at p ≤ 8 for `y_blocks`, `supplement` and `blocks`, 1e-5.
#[test]
fn f32_agrees_with_f64_at_low_degree() {
    let p = 8;
    for beta in angles() {
        let (s, x) = (beta as f32).sin_cos();
        let mut single = vec![0.0f32; compact_blocks_len(p)];
        y_blocks(p, x, s, &mut single, &mut Scratch::new(p));
        let double = y(p, beta);
        let worst = single
            .iter()
            .zip(&double)
            .fold(0.0f64, |m, (&a, b)| m.max((a as f64 - b).abs()));
        assert!(worst <= 1e-5, "y_blocks, β = {beta}: {worst:e}");
        supplement(p, &mut single);
        let mut signed = double.clone();
        supplement(p, &mut signed);
        let worst = single
            .iter()
            .zip(&signed)
            .fold(0.0f64, |m, (&a, b)| m.max((a as f64 - b).abs()));
        assert!(worst <= 1e-5, "supplement, β = {beta}: {worst:e}");
    }
    let (alpha, beta, gamma) = (0.4, 1.1, -2.3);
    let mut single = vec![0.0f32; blocks_len(p)];
    blocks(
        p,
        alpha as f32,
        beta as f32,
        gamma as f32,
        &mut single,
        &mut Scratch::new(p),
    );
    let mut double = vec![0.0f64; blocks_len(p)];
    blocks(p, alpha, beta, gamma, &mut double, &mut Scratch::new(p));
    let worst = single
        .iter()
        .zip(&double)
        .fold(0.0f64, |m, (&a, b)| m.max((a as f64 - b).abs()));
    assert!(worst <= 1e-5, "blocks: {worst:e}");
}

/// Scratch for a smaller degree panics.
#[test]
#[should_panic(expected = "scratch is for degree")]
fn small_scratch_panics() {
    let mut out = vec![0.0; blocks_len(4)];
    blocks(4, 0.1, 0.2, 0.3, &mut out, &mut Scratch::new(3));
}

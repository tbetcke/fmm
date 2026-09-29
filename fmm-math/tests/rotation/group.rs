//! Group properties of the blocks (CONVENTIONS §3.8): the identity, the homomorphism
//! Dⁿ(Q₁Q₂) = Dⁿ(Q₁) Dⁿ(Q₂), and orthogonality of N Dⁿ N⁻¹.
//!
//! Entries of Dⁿ itself range over many orders of magnitude, from Nⱼ/Nᵢ; the
//! comparisons are therefore made on the orthonormal blocks N Dⁿ N⁻¹, whose entries are
//! bounded by one, with absolute errors.

use nd_fmm_math::rotation::{block_range, blocks, blocks_len};

use crate::common::SplitMix64;
use crate::support::{
    Matrix, mat_mul, max_abs_diff, orthonormal_block, random_rotation, square_mul,
};

const P: usize = 30;

/// Error bound for degree n: 1e-13 for n ≤ 20 and 1e-11 above, as in the brief's
/// action test.
fn tolerance(n: usize) -> f64 {
    if n <= 20 { 1e-13 } else { 1e-11 }
}

fn compute(q: &Matrix) -> Vec<f64> {
    let mut d = vec![0.0; blocks_len(P)];
    blocks(P, q, &mut d);
    d
}

#[test]
fn identity_maps_to_identity() {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let d = compute(&identity);
    for n in 0..=P {
        let width = 2 * n + 1;
        for (e, &v) in d[block_range(n)].iter().enumerate() {
            let expected = if e / width == e % width { 1.0 } else { 0.0 };
            assert_eq!(v, expected, "degree {n}, entry {e}");
        }
    }
}

#[test]
fn blocks_are_a_homomorphism() {
    let mut rng = SplitMix64::new(0x4040_0e01);
    let mut worst = [0.0_f64; P + 1];
    for _ in 0..20 {
        let q1 = random_rotation(&mut rng);
        let q2 = random_rotation(&mut rng);
        let (d1, d2, d12) = (compute(&q1), compute(&q2), compute(&mat_mul(&q1, &q2)));
        for (n, w) in worst.iter_mut().enumerate() {
            let product = square_mul(
                2 * n + 1,
                &orthonormal_block(&d1, n),
                &orthonormal_block(&d2, n),
            );
            *w = w.max(max_abs_diff(&orthonormal_block(&d12, n), &product));
        }
    }
    println!(
        "homomorphism: worst error by degree {}",
        worst.map(|w| format!("{w:.1e}")).join(" ")
    );
    for (n, &w) in worst.iter().enumerate() {
        assert!(w <= tolerance(n), "degree {n}: {w:e} > {:e}", tolerance(n));
    }
}

/// The brief requires 1e-13 for p ≤ 20; the design document (C0.3 in
/// docs/design/laplace-fmm-plan.md) asks for it up to p = 30, so every degree is
/// checked at 1e-13.
#[test]
fn normalised_blocks_are_orthogonal() {
    let mut rng = SplitMix64::new(0x4040_0e02);
    let mut worst = [0.0_f64; P + 1];
    for _ in 0..20 {
        let d = compute(&random_rotation(&mut rng));
        for (n, w) in worst.iter_mut().enumerate() {
            let width = 2 * n + 1;
            let b = orthonormal_block(&d, n);
            let mut transpose = vec![0.0; b.len()];
            for (e, &v) in b.iter().enumerate() {
                transpose[(e % width) * width + e / width] = v;
            }
            let product = square_mul(width, &b, &transpose);
            let identity: Vec<f64> = (0..b.len())
                .map(|e| if e / width == e % width { 1.0 } else { 0.0 })
                .collect();
            *w = w.max(max_abs_diff(&product, &identity));
        }
    }
    println!(
        "orthogonality: worst error by degree {}",
        worst.map(|w| format!("{w:.1e}")).join(" ")
    );
    for (n, &w) in worst.iter().enumerate() {
        assert!(w <= 1e-13, "degree {n}: {w:e} > 1e-13");
    }
}

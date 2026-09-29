//! Helpers shared by the rotation tests: random rotations, 3 × 3 products, block
//! application and the normalisation N of CONVENTIONS §3.8.

use nd_fmm_math::Layout;
use nd_fmm_math::rotation::block_range;

use crate::common::SplitMix64;

/// A 3 × 3 matrix, row-major: `q[i][j]` = Qᵢⱼ.
pub type Matrix = [[f64; 3]; 3];

/// Uniformly distributed rotation, from a uniformly distributed unit quaternion
/// (rejection sampling in the unit 4-ball).
pub fn random_rotation(rng: &mut SplitMix64) -> Matrix {
    let [w, x, y, z] = loop {
        let v = [0; 4].map(|_| rng.range(-1.0, 1.0));
        let r = v.iter().map(|c| c * c).sum::<f64>().sqrt();
        if (1e-3..=1.0).contains(&r) {
            break v.map(|c| c / r);
        }
    };
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

pub fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = [[0.0; 3]; 3];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

pub fn mat_vec(q: &Matrix, x: [f64; 3]) -> [f64; 3] {
    q.map(|row| row[0] * x[0] + row[1] * x[1] + row[2] * x[2])
}

/// Rotation about z by `angle`.
pub fn rot_z(angle: f64) -> Matrix {
    let (s, c) = angle.sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// Rotation about y by `angle`.
pub fn rot_y(angle: f64) -> Matrix {
    let (s, c) = angle.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Dⁿ vₙ for every degree n ≤ p, with the blocks in the row-major layout of
/// `rotation::blocks` and v in real storage (CONVENTIONS §3.6).
pub fn apply(p: usize, blocks: &[f64], v: &[f64]) -> Vec<f64> {
    let layout = Layout::new(p);
    let mut out = vec![0.0; layout.len()];
    for (n, range) in layout.degrees() {
        let block = &blocks[block_range(n)];
        for (row, o) in block.chunks_exact(2 * n + 1).zip(&mut out[range.clone()]) {
            *o = row.iter().zip(&v[range.clone()]).map(|(d, x)| d * x).sum();
        }
    }
    out
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product()
}

/// Nₘ = √((n + |m|)! (n − |m|)!) · cₘ of CONVENTIONS §3.8, c₀ = 1 and cₘ = √2
/// otherwise.
pub fn normalisation(n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 { 1.0 } else { 2.0_f64.sqrt() };
    (factorial(n + k) * factorial(n - k)).sqrt() * c
}

/// Nₘ / Sₘ = cₘ / √((n + |m|)! (n − |m|)!), with S = diag((n − |m|)! (n + |m|)!) of
/// CONVENTIONS §3.8: the weight that takes irregular harmonics in real storage to the
/// orthonormal basis, in which the irregular blocks S Dⁿ S⁻¹ become N Dⁿ N⁻¹.
pub fn orthonormal_weight(n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    normalisation(n, m) / (factorial(n + k) * factorial(n - k))
}

/// N Dⁿ N⁻¹ of degree n, row-major; orthogonal for a rotation (CONVENTIONS §3.8).
pub fn orthonormal_block(blocks: &[f64], n: usize) -> Vec<f64> {
    let width = 2 * n + 1;
    let order = |i: usize| i as isize - n as isize;
    blocks[block_range(n)]
        .iter()
        .enumerate()
        .map(|(e, d)| {
            let (i, j) = (order(e / width), order(e % width));
            d * normalisation(n, i) / normalisation(n, j)
        })
        .collect()
}

/// Product of two row-major square matrices of width `width`.
pub fn square_mul(width: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; width * width];
    for i in 0..width {
        for k in 0..width {
            let aik = a[i * width + k];
            for j in 0..width {
                c[i * width + j] += aik * b[k * width + j];
            }
        }
    }
    c
}

/// Largest absolute entry of a − b.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0_f64, |acc, (x, y)| acc.max((x - y).abs()))
}

//! Rotation blocks for real solid harmonics (CONVENTIONS §3.8).
//!
//! A proper rotation Q leaves every degree n invariant: for each n there is a real
//! (2n + 1) × (2n + 1) block Dⁿ(Q) with
//!
//! Rₙ(Qx) = Dⁿ(Q) Rₙ(x),  Dⁿ(Q₁Q₂) = Dⁿ(Q₁) Dⁿ(Q₂),
//!
//! where Rₙ is the vector of the 2n + 1 regular harmonics of degree n in the real
//! storage of CONVENTIONS §3.6. The same blocks conjugated by
//! S = diag((n − |m|)! (n + |m|)!) rotate irregular harmonics (see [`to_irregular`]),
//! and N Dⁿ N⁻¹ with N = diag(√((n + |m|)! (n − |m|)!) · cₘ), c₀ = 1 and cₘ = √2
//! otherwise, is orthogonal.
//!
//! # Block layout
//!
//! [`blocks`] writes the blocks of degrees 0 to p contiguously, degree by degree, in
//! [`blocks_len`]`(p)` = Σₙ (2n + 1)² = (p + 1)(2p + 1)(2p + 3)/3 slots; degree n
//! occupies [`block_range`]`(n)`, which does not depend on p. Each block is row-major,
//! with rows and columns in the order of the real storage within a degree, m = −n to n:
//! entry (i, j), −n ≤ i, j ≤ n, is at offset (i + n)(2n + 1) + (j + n) of its block and
//! maps slot j of Rₙ(x) to slot i of Rₙ(Qx).
//!
//! # Method
//!
//! The blocks are built degree by degree from the chain rule, as in the recursion of
//! Ivanic and Ruedenberg (J. Phys. Chem. 100, 6342 (1996); erratum 102, 9099 (1998)),
//! here written for the Racah-normalised harmonics of CONVENTIONS §3.3, where the
//! gradient ladder of §3.4 has the coefficients 1 and 1/2. In complex form, with
//! Rₙᵐ(Qx) = Σₖ 𝒟ₘₖ Rₙᵏ(x) and D± = ∂x ± i∂y, a real direction vector t acts on
//! y = Qx as
//!
//! Σᵢ tᵢ ∂/∂yᵢ = ½(t₁ + i t₂) D− + ½(t₁ − i t₂) D+ + t₃ ∂z,
//!
//! and D− Rₙᵐ = Rₙ₋₁ᵐ⁻¹, D+ Rₙᵐ = −Rₙ₋₁ᵐ⁺¹, ∂z Rₙᵐ = Rₙ₋₁ᵐ (§3.4). Applying ∂z, D−
//! and D+ in the x frame to both sides of the definition, which in the y frame are
//! Σᵢ Qᵢ₃ ∂/∂yᵢ and Σᵢ (Qᵢ₁ ∓ i Qᵢ₂) ∂/∂yᵢ, and comparing coefficients of the linearly
//! independent Rₙ₋₁ˡ(x) gives, for 0 ≤ m ≤ n,
//!
//! - |k| ≤ n − 1: 𝒟ⁿₘₖ = a 𝒟ⁿ⁻¹ₘ₋₁,ₖ − b 𝒟ⁿ⁻¹ₘ₊₁,ₖ + Q₃₃ 𝒟ⁿ⁻¹ₘ,ₖ, with
//!   a = ½(Q₁₃ + i Q₂₃) and b = ½(Q₁₃ − i Q₂₃);
//! - k = n: the same with (Qᵢ₁ − i Qᵢ₂) in place of Qᵢ₃ and column n − 1 of 𝒟ⁿ⁻¹;
//! - k = −n: minus the same with (Qᵢ₁ + i Qᵢ₂) and column −(n − 1),
//!
//! where 𝒟ⁿ⁻¹ⱼₗ = 0 for |j| > n − 1, and the rows m < 0 follow from
//! 𝒟₋ₘ,₋ₖ = (−1)ᵐ⁺ᵏ conj(𝒟ₘₖ), which is the symmetry Xₙ⁻ᵐ = (−1)ᵐ conj(Xₙᵐ) of
//! §3.3. Only the real blocks are stored; complex entries of degree n − 1 are recovered
//! from the previous block when needed. Up to the diagonal scaling N this is the
//! Ivanic–Ruedenberg recursion, so it shares its stability; the tests confirm the
//! accuracy of CONVENTIONS §3.9 up to p = 30 in f64.
//!
//! No allocation takes place: each degree is computed from the previous one inside the
//! caller's slice.

use core::ops::Range;

use crate::RealScalar;

/// A complex number as (real part, imaginary part).
type Complex<T> = (T, T);

#[inline(always)]
fn add<T: RealScalar>((a, b): Complex<T>, (c, d): Complex<T>) -> Complex<T> {
    (a + c, b + d)
}

#[inline(always)]
fn sub<T: RealScalar>((a, b): Complex<T>, (c, d): Complex<T>) -> Complex<T> {
    (a - c, b - d)
}

#[inline(always)]
fn mul<T: RealScalar>((a, b): Complex<T>, (c, d): Complex<T>) -> Complex<T> {
    (a * c - b * d, a * d + b * c)
}

#[inline(always)]
fn scale<T: RealScalar>(s: T, (a, b): Complex<T>) -> Complex<T> {
    (s * a, s * b)
}

/// The integer `k` as a `T`; exact for every k that occurs here.
#[inline(always)]
fn int<T: RealScalar>(k: usize) -> T {
    T::from_f64(k as f64)
}

/// (−1)ᵏ as a `T`.
#[inline(always)]
fn parity<T: RealScalar>(k: isize) -> T {
    if k % 2 == 0 { T::one() } else { -T::one() }
}

/// Offset of the degree-n block, Σₖ₍ₖ₌₀..ₙ₋₁₎ (2k + 1)² = n(2n − 1)(2n + 1)/3.
#[inline]
const fn block_offset(n: usize) -> usize {
    (4 * n * n * n - n) / 3
}

/// Number of slots of the blocks of degrees 0 to `p`, Σₙ (2n + 1)² =
/// (p + 1)(2p + 1)(2p + 3)/3.
///
/// ```
/// use nd_fmm_math::rotation::blocks_len;
///
/// assert_eq!(blocks_len(0), 1);
/// assert_eq!(blocks_len(2), 1 + 9 + 25);
/// ```
#[inline]
pub const fn blocks_len(p: usize) -> usize {
    block_offset(p + 1)
}

/// Range of the degree-n block Dⁿ in the output of [`blocks`]; it does not depend on p.
/// The block is row-major with rows and columns in the order m = −n to n (see the
/// [module documentation](self)).
///
/// ```
/// use nd_fmm_math::rotation::block_range;
///
/// assert_eq!(block_range(0), 0..1);
/// assert_eq!(block_range(2), 10..35);
/// ```
#[inline]
pub const fn block_range(n: usize) -> Range<usize> {
    block_offset(n)..block_offset(n + 1)
}

/// Index of entry (i, j), −n ≤ i, j ≤ n, within a row-major block of degree n.
#[inline(always)]
fn entry(n: usize, i: isize, j: isize) -> usize {
    let n = n as isize;
    ((i + n) * (2 * n + 1) + (j + n)) as usize
}

/// Complex matrix element 𝒟ⱼₗ of degree d, for any j and |l| ≤ d, recovered from the
/// real block `block` of that degree; zero for |j| > d.
///
/// For m > 0 and k > 0, write A = 𝒟ₘₖ and B = (−1)ᵏ 𝒟ₘ,₋ₖ. Then row +m of the real
/// block holds Re(A + B) in column +k and −Im(A − B) in column −k, row −m holds
/// Im(A + B) and Re(A − B), and column 0 holds Re 𝒟ₘ₀ and Im 𝒟ₘ₀ (see [`blocks`]).
/// Row 0 holds only the first of each pair, the other vanishing.
fn complex_entry<T: RealScalar>(block: &[T], d: usize, j: isize, l: isize) -> Complex<T> {
    let zero = T::zero();
    if j.unsigned_abs() > d {
        return (zero, zero);
    }
    if j < 0 {
        // 𝒟₋ₘ,₋ₖ = (−1)ᵐ⁺ᵏ conj(𝒟ₘₖ)
        let (re, im) = complex_entry(block, d, -j, -l);
        return scale(parity(j + l), (re, -im));
    }
    let at = |i: isize, k: isize| block[entry(d, i, k)];
    if l == 0 {
        return if j == 0 {
            (at(0, 0), zero)
        } else {
            (at(j, 0), at(-j, 0))
        };
    }
    let k = l.abs();
    let (sum, diff) = if j == 0 {
        ((at(0, k), zero), (zero, -at(0, -k)))
    } else {
        ((at(j, k), at(-j, k)), (at(-j, -k), -at(j, -k)))
    };
    let half = T::from_f64(0.5);
    if l > 0 {
        scale(half, add(sum, diff))
    } else {
        scale(half * parity(k), sub(sum, diff))
    }
}

/// The action of a direction vector t in the y frame on Rₙᵐ, Σᵢ tᵢ ∂/∂yᵢ Rₙᵐ =
/// `lower` Rₙ₋₁ᵐ⁻¹ − `raise` Rₙ₋₁ᵐ⁺¹ + `keep` Rₙ₋₁ᵐ, from the ladder of CONVENTIONS
/// §3.4.
struct Ladder<T> {
    lower: Complex<T>,
    raise: Complex<T>,
    keep: Complex<T>,
}

impl<T: RealScalar> Ladder<T> {
    /// Ladder of the complex direction t: lower = ½(t₁ + i t₂), raise = ½(t₁ − i t₂),
    /// keep = t₃.
    fn new([t1, t2, t3]: [Complex<T>; 3]) -> Self {
        let half = T::from_f64(0.5);
        let i_t2 = (-t2.1, t2.0);
        Self {
            lower: scale(half, add(t1, i_t2)),
            raise: scale(half, sub(t1, i_t2)),
            keep: t3,
        }
    }

    /// Coefficient of Rₙ₋₁ˡ(x) in the action on Rₙᵐ(Qx), from the block of degree
    /// d = n − 1: lower 𝒟ₘ₋₁,ₗ − raise 𝒟ₘ₊₁,ₗ + keep 𝒟ₘ,ₗ.
    #[inline]
    fn apply(&self, previous: &[T], d: usize, m: isize, l: isize) -> Complex<T> {
        let lowered = mul(self.lower, complex_entry(previous, d, m - 1, l));
        let raised = mul(self.raise, complex_entry(previous, d, m + 1, l));
        let kept = mul(self.keep, complex_entry(previous, d, m, l));
        add(sub(lowered, raised), kept)
    }
}

/// Rotation blocks Dⁿ(Q), n ≤ p, of the proper rotation `q` (CONVENTIONS §3.8),
/// written to `out` degree by degree, each block row-major (see the
/// [module documentation](self)).
///
/// `q[i][j]` is Qᵢⱼ, so that (Qx)ᵢ = Σⱼ `q[i][j]` xⱼ. The blocks satisfy
/// Rₙ(Qx) = Dⁿ(Q) Rₙ(x) in real storage (CONVENTIONS §3.6); use [`to_irregular`] to
/// rotate irregular harmonics. `q` must be orthogonal with determinant +1; this is
/// not checked. Blocks of degree n do not depend on p.
///
/// ```
/// use nd_fmm_math::rotation::{block_range, blocks, blocks_len, euler_zyz};
/// use nd_fmm_math::{Layout, harmonics};
///
/// let p = 4;
/// let q = euler_zyz(0.3, 1.1, -0.4);
/// let mut d = vec![0.0; blocks_len(p)];
/// blocks(p, &q, &mut d);
///
/// let x = [0.2, -0.5, 0.7];
/// let qx = q.map(|row: [f64; 3]| row[0] * x[0] + row[1] * x[1] + row[2] * x[2]);
/// let layout = Layout::new(p);
/// let (mut r, mut rq) = (vec![0.0; layout.len()], vec![0.0; layout.len()]);
/// harmonics::regular(p, x, &mut r);
/// harmonics::regular(p, qx, &mut rq);
///
/// // Rₙ(Qx) = Dⁿ(Q) Rₙ(x) at degree n = 3
/// let n = 3;
/// let rows = d[block_range(n)].chunks_exact(2 * n + 1);
/// for (row, expected) in rows.zip(&rq[layout.degree(n)]) {
///     let value: f64 = row.iter().zip(&r[layout.degree(n)]).map(|(a, b)| a * b).sum();
///     assert!((value - expected).abs() < 1e-14);
/// }
/// ```
///
/// # Panics
///
/// If `out.len() != blocks_len(p)`.
pub fn blocks<T: RealScalar>(p: usize, q: &[[T; 3]; 3], out: &mut [T]) {
    assert_eq!(
        out.len(),
        blocks_len(p),
        "`out` must have length blocks_len(p) = {} for p = {p}",
        blocks_len(p)
    );
    let zero = T::zero();
    // ∂z, D− = ∂x − i∂y and D+ = ∂x + i∂y in the x frame, as directions in the y frame.
    let dz = Ladder::new(q.map(|row| (row[2], zero)));
    let lower = Ladder::new(q.map(|row| (row[0], -row[1])));
    let raise = Ladder::new(q.map(|row| (row[0], row[1])));
    out[0] = T::one();
    for n in 1..=p {
        let (done, rest) = out.split_at_mut(block_offset(n));
        let previous = &done[block_range(n - 1)];
        let block = &mut rest[..(2 * n + 1) * (2 * n + 1)];
        let (d, top) = (n - 1, n as isize);
        for m in 0..=top {
            // 𝒟ⁿₘₖ for any −n ≤ k ≤ n
            let element = |k: isize| {
                if k.unsigned_abs() <= d {
                    dz.apply(previous, d, m, k)
                } else if k > 0 {
                    lower.apply(previous, d, m, k - 1)
                } else {
                    scale(-T::one(), raise.apply(previous, d, m, k + 1))
                }
            };
            let (re, im) = element(0);
            block[entry(n, m, 0)] = re;
            if m > 0 {
                block[entry(n, -m, 0)] = im;
            }
            for k in 1..=top {
                // With A = 𝒟ₘₖ, B = (−1)ᵏ 𝒟ₘ,₋ₖ and Rₙᵏ(x) = u + iv, the two terms of
                // ±k contribute A Rₙᵏ + B conj(Rₙᵏ) = (A + B) u + i (A − B) v.
                let a = element(k);
                let b = scale(parity(k), element(-k));
                let (sum, diff) = (add(a, b), sub(a, b));
                block[entry(n, m, k)] = sum.0;
                block[entry(n, m, -k)] = -diff.1;
                if m > 0 {
                    block[entry(n, -m, k)] = sum.1;
                    block[entry(n, -m, -k)] = diff.0;
                }
            }
        }
    }
}

/// Converts the regular blocks Dⁿ, n ≤ p, from [`blocks`] in place into the blocks
/// S Dⁿ S⁻¹ that rotate irregular harmonics, Iₙ(Qx) = S Dⁿ(Q) S⁻¹ Iₙ(x), with
/// S = diag((n − |m|)! (n + |m|)!) (CONVENTIONS §3.8).
///
/// This holds because Iₙᵐ = (n − m)! (n + m)! Rₙᵐ / r²ⁿ⁺¹ and |Qx| = |x|. Entry (i, j)
/// is multiplied by w(|i|) / w(|j|) with w(k) = Sₖ / S₀ = Πₜ₌₁..ₖ (n + t)/(n − t + 1),
/// so no factorial is formed.
///
/// ```
/// use nd_fmm_math::rotation::{block_range, blocks, blocks_len, euler_zyz, to_irregular};
/// use nd_fmm_math::{Layout, harmonics};
///
/// let p = 4;
/// let q = euler_zyz(-1.2, 0.8, 2.5);
/// let mut d = vec![0.0; blocks_len(p)];
/// blocks(p, &q, &mut d);
/// to_irregular(p, &mut d);
///
/// let x = [2.0, -1.5, 0.5];
/// let qx = q.map(|row: [f64; 3]| row[0] * x[0] + row[1] * x[1] + row[2] * x[2]);
/// let layout = Layout::new(p);
/// let (mut i, mut iq) = (vec![0.0; layout.len()], vec![0.0; layout.len()]);
/// harmonics::irregular(p, x, &mut i);
/// harmonics::irregular(p, qx, &mut iq);
///
/// // Iₙ(Qx) = S Dⁿ(Q) S⁻¹ Iₙ(x) at degree n = 4
/// let n = 4;
/// let rows = d[block_range(n)].chunks_exact(2 * n + 1);
/// for (row, expected) in rows.zip(&iq[layout.degree(n)]) {
///     let value: f64 = row.iter().zip(&i[layout.degree(n)]).map(|(a, b)| a * b).sum();
///     assert!((value - expected).abs() < 1e-15);
/// }
/// ```
///
/// # Panics
///
/// If `blocks.len() != blocks_len(p)`.
pub fn to_irregular<T: RealScalar>(p: usize, blocks: &mut [T]) {
    assert_eq!(
        blocks.len(),
        blocks_len(p),
        "`blocks` must have length blocks_len(p) = {} for p = {p}",
        blocks_len(p)
    );
    for n in 1..=p {
        let block = &mut blocks[block_range(n)];
        let width = 2 * n + 1;
        let mut w = T::one();
        for k in 1..=n {
            w = w * int(n + k) / int(n - k + 1);
            let k = k as isize;
            for m in [k, -k] {
                let i = (m + n as isize) as usize;
                for v in &mut block[i * width..(i + 1) * width] {
                    *v = *v * w;
                }
                for v in block[i..].iter_mut().step_by(width) {
                    *v = *v / w;
                }
            }
        }
    }
}

/// Rotation matrix Q = R_z(α) R_y(β) R_z(γ) of the z-y-z Euler angles α, β, γ
/// (CONVENTIONS §3.8), row-major as expected by [`blocks`], with
///
/// R_z(φ) = [cos φ, −sin φ, 0; sin φ, cos φ, 0; 0, 0, 1],
/// R_y(φ) = [cos φ, 0, sin φ; 0, 1, 0; −sin φ, 0, cos φ].
///
/// Q maps the z axis to the direction with polar angle β and azimuth α.
///
/// ```
/// use nd_fmm_math::rotation::euler_zyz;
///
/// let (alpha, beta) = (0.4_f64, 1.2_f64);
/// let q = euler_zyz(alpha, beta, -0.7);
/// let axis = [q[0][2], q[1][2], q[2][2]];
/// let expected = [beta.sin() * alpha.cos(), beta.sin() * alpha.sin(), beta.cos()];
/// for (a, e) in axis.iter().zip(expected) {
///     assert!((a - e).abs() < 1e-15);
/// }
/// ```
pub fn euler_zyz<T: RealScalar>(alpha: T, beta: T, gamma: T) -> [[T; 3]; 3] {
    let (sa, ca) = alpha.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let (sg, cg) = gamma.sin_cos();
    [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute(p: usize, q: &[[f64; 3]; 3]) -> Vec<f64> {
        let mut out = vec![f64::NAN; blocks_len(p)];
        blocks(p, q, &mut out);
        out
    }

    #[test]
    fn block_ranges_tile_the_output() {
        let mut end = 0;
        for n in 0..40 {
            let range = block_range(n);
            assert_eq!(range.start, end);
            assert_eq!(range.len(), (2 * n + 1) * (2 * n + 1));
            end = range.end;
            assert_eq!(blocks_len(n), end);
        }
    }

    /// In real storage R₁ = (Im R₁¹, R₁⁰, Re R₁¹) = (y/2, z, x/2) (CONVENTIONS §3.3), so
    /// D¹ = C P Q Pᵀ C⁻¹ with P the permutation (x, y, z) ↦ (y, z, x) and
    /// C = diag(1/2, 1, 1/2).
    #[test]
    fn degree_one_block_is_the_permuted_rotation() {
        let q = euler_zyz(0.7, -1.3, 2.1);
        let d = compute(1, &q);
        let (axis, c) = ([1, 2, 0], [0.5, 1.0, 0.5]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = c[i] * q[axis[i]][axis[j]] / c[j];
                assert!((d[1 + 3 * i + j] - expected).abs() < 1e-15, "({i}, {j})");
            }
        }
    }

    #[test]
    fn lower_blocks_do_not_depend_on_p() {
        let q = euler_zyz(-0.3, 2.2, 0.9);
        let full = compute(12, &q);
        for p in 0..12 {
            assert_eq!(compute(p, &q), full[..blocks_len(p)]);
        }
    }

    #[test]
    #[should_panic(expected = "must have length blocks_len(p)")]
    fn wrong_output_length_panics() {
        blocks(3, &euler_zyz(0.0, 0.0, 0.0), &mut [0.0; 83]);
    }
}

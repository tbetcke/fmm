//! The defining property of the blocks, Rₙ(Qx) = Dⁿ(Q) Rₙ(x), and its irregular
//! counterpart Iₙ(Qx) = S Dⁿ(Q) S⁻¹ Iₙ(x) (CONVENTIONS §3.8).

use nd_fmm_math::harmonics::{irregular, regular};
use nd_fmm_math::rotation::{blocks, blocks_len, to_irregular};
use nd_fmm_math::{Layout, RealScalar};

use crate::common::{SplitMix64, degree_relative_error};
use crate::support::{apply, mat_vec, orthonormal_weight, random_rotation};

type Eval<S> = fn(usize, [S; 3], &mut [S]);

#[derive(Clone, Copy, Debug)]
enum Family {
    Regular,
    Irregular,
}

/// Checks Xₙ(Qx) against the rotated Xₙ(x) at `samples` random points and rotations,
/// for every degree n ≤ p; `tolerances` lists pairs (pₖ, tolₖ), and the degrees
/// n ≤ pₖ must reach the degree-relative error tolₖ.
///
/// Blocks and harmonics at x are computed in T, from Q and x rounded to T; the
/// reference Xₙ(Qx) is computed in f64 from the same rounded Q and x, and the block
/// product is formed in f64.
///
/// # Error measure
///
/// Regular harmonics are compared in real storage, as in the harmonics tests. For
/// irregular harmonics that measure is ill-conditioned: the entries of S Dⁿ S⁻¹ span
/// up to (2n)!/(n!)² ≈ 1.2e17 at n = 30, so a rotation can map a vector whose large
/// entries sit at high |m| to one concentrated at small |m|, and the product cancels.
/// The condition number max Σⱼ |(S D S⁻¹)ᵢⱼ Iⱼ| / max |Iᵢ(Qx)| of that measure
/// reaches 3e8 at n = 30, so even exact blocks applied to correctly rounded Iₙ(x)
/// would miss 1e-13 by orders of magnitude. Irregular harmonics are therefore compared
/// in the orthonormal basis, slot m weighted by Nₘ / Sₘ (CONVENTIONS §3.8), in which
/// the rotation is the orthogonal N Dⁿ N⁻¹ and the measure is well conditioned.
fn check_action<T: RealScalar>(
    seed: u64,
    family: Family,
    p: usize,
    samples: usize,
    tolerances: &[(usize, f64)],
) {
    let (harmonic, harmonic_f64, rmin, rmax): (Eval<T>, Eval<f64>, f64, f64) = match family {
        // CONVENTIONS §3.9: regular harmonics at |x| ≤ √3, irregular at |x| ≥ 2.
        Family::Regular => (regular, regular, 0.1, 1.8),
        Family::Irregular => (irregular, irregular, 2.0, 8.0),
    };
    let len = Layout::new(p).len();
    let mut rng = SplitMix64::new(seed);
    let mut d = vec![T::zero(); blocks_len(p)];
    let mut hx = vec![T::zero(); len];
    let mut hy = vec![0.0; len];
    let mut worst = vec![0.0_f64; tolerances.len()];
    for _ in 0..samples {
        let q = random_rotation(&mut rng).map(|row| row.map(T::from_f64));
        let x = rng.point_in_shell(rmin, rmax).map(T::from_f64);
        blocks(p, &q, &mut d);
        if let Family::Irregular = family {
            to_irregular(p, &mut d);
        }
        harmonic(p, x, &mut hx);

        let q64 = q.map(|row| row.map(|c| c.to_f64()));
        let y = mat_vec(&q64, x.map(|c| c.to_f64()));
        harmonic_f64(p, y, &mut hy);

        let d64: Vec<f64> = d.iter().map(|&v| v.to_f64()).collect();
        let hx64: Vec<f64> = hx.iter().map(|&v| v.to_f64()).collect();
        let mut rotated = apply(p, &d64, &hx64);
        if let Family::Irregular = family {
            for (i, (r, h)) in rotated.iter_mut().zip(&mut hy).enumerate() {
                let (n, m) = Layout::new(p).nm(i);
                *r *= orthonormal_weight(n, m);
                *h *= orthonormal_weight(n, m);
            }
        }
        for ((pk, _), w) in tolerances.iter().zip(&mut worst) {
            let l = Layout::new(*pk).len();
            *w = w.max(degree_relative_error(*pk, &rotated[..l], &hy[..l]));
        }
    }
    for ((pk, tol), w) in tolerances.iter().zip(&worst) {
        println!("{family:?}, n <= {pk}: worst degree-relative error {w:.2e}");
        assert!(w <= tol, "{family:?}, n <= {pk}: {w:e} > {tol:e}");
    }
}

/// The brief: relative 1e-13 for p ≤ 20 and 1e-11 for p ≤ 30.
const F64_TOLERANCES: [(usize, f64); 2] = [(20, 1e-13), (30, 1e-11)];

#[test]
fn regular_harmonics_rotate_with_blocks() {
    check_action::<f64>(0x0707_a7e1, Family::Regular, 30, 100, &F64_TOLERANCES);
}

#[test]
fn irregular_harmonics_rotate_with_conjugated_blocks() {
    check_action::<f64>(0x0707_a7e2, Family::Irregular, 30, 100, &F64_TOLERANCES);
}

/// Not required by the brief: the same property in f32, p ≤ 8 (§3.9).
#[test]
fn regular_harmonics_rotate_with_blocks_f32() {
    check_action::<f32>(0x0707_a7e3, Family::Regular, 8, 100, &[(8, 1e-5)]);
}

/// Not required by the brief: the same property in f32, p ≤ 8 (§3.9).
#[test]
fn irregular_harmonics_rotate_with_conjugated_blocks_f32() {
    check_action::<f32>(0x0707_a7e4, Family::Irregular, 8, 100, &[(8, 1e-5)]);
}

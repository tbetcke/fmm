//! The portable scalar path: a plain loop with `sqrt` and division, written for
//! clarity, not speed (docs/design/simd-p2p.md §5.2).
//!
//! For each target, the sources are visited in input order and each term is added
//! directly into the outputs, with the formulas and operation order of
//! `nd_fmm_ref::p2p::p2p`: d = x − y, r² = (d₀² + d₁²) + d₂², r = √r², then
//! φ ← φ + q / r and (∇φ)ₖ ← (∇φ)ₖ − (q dₖ) / (r² r). Rust never contracts a
//! multiplication and an addition into an fma on its own, so on pairs with r² ≠ 0 the
//! result is the reference's bit for bit. A pair with r² = 0 is skipped
//! (CONVENTIONS §3.13, "Fast kernels"), where the reference skips x == y.

use nd_fmm_math::RealScalar;

/// P2P with the scalar path, with lengths already checked by the caller.
pub(crate) fn p2p<T: RealScalar>(
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    match gradient {
        None => {
            for (&x, phi) in targets.iter().zip(potential.iter_mut()) {
                for (&y, &q) in sources.iter().zip(charges) {
                    let (_, r2) = separation(x, y);
                    if r2 == T::zero() {
                        continue;
                    }
                    *phi = *phi + q / r2.sqrt();
                }
            }
        }
        Some(gradient) => {
            for ((&x, phi), g) in targets.iter().zip(potential.iter_mut()).zip(gradient) {
                for (&y, &q) in sources.iter().zip(charges) {
                    let (d, r2) = separation(x, y);
                    if r2 == T::zero() {
                        continue;
                    }
                    let r = r2.sqrt();
                    let r3 = r2 * r;
                    *phi = *phi + q / r;
                    for (gk, dk) in g.iter_mut().zip(d) {
                        *gk = *gk + -(q * dk / r3);
                    }
                }
            }
        }
    }
}

/// d = x − y and r² = (d₀² + d₁²) + d₂².
#[inline]
fn separation<T: RealScalar>(x: [T; 3], y: [T; 3]) -> ([T; 3], T) {
    let d = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    (d, d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

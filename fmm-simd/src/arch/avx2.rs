//! x86_64 AVX2 + FMA (256-bit vectors: 8 × f32, 4 × f64).
//!
//! Placeholder until Phase 3S T4 and T5: the entry point runs the scalar path. The
//! vector code will live in `#[target_feature(enable = "avx2,fma")]` functions, called
//! only after [`Isa::Avx2`](crate::Isa::Avx2) was detected.

use crate::SimdScalar;

/// P2P on AVX2 + FMA, with lengths already checked by the caller; for now the scalar
/// path.
pub(crate) fn p2p<T: SimdScalar>(
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    super::scalar::p2p(sources, charges, targets, potential, gradient);
}

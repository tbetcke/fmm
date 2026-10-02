//! aarch64 NEON (128-bit vectors: 4 × f32, 2 × f64).
//!
//! Placeholder until Phase 3S T4 and T5: the entry point runs the scalar path. The
//! vector code will live in `#[target_feature(enable = "neon")]` functions. NEON is
//! enabled for every aarch64 target, but on the pinned toolchain an intrinsic is safe to
//! call only inside a function that lists `neon` in its own `#[target_feature]`, so the
//! dispatch will call the entry point in an `unsafe` block (crate documentation,
//! "Toolchain").

use crate::SimdScalar;

/// P2P on NEON, with lengths already checked by the caller; for now the scalar path.
pub(crate) fn p2p<T: SimdScalar>(
    sources: &[[T; 3]],
    charges: &[T],
    targets: &[[T; 3]],
    potential: &mut [T],
    gradient: Option<&mut [[T; 3]]>,
) {
    super::scalar::p2p(sources, charges, targets, potential, gradient);
}

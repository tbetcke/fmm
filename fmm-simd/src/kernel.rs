//! The P2P kernel bound to one ISA (docs/design/simd-p2p.md §5.1 and §5.2).

use std::marker::PhantomData;

use crate::{Isa, IsaUnavailable, SimdScalar, arch};

/// A P2P kernel bound to one available [`Isa`].
///
/// [`evaluate`](Self::evaluate) has the signature and semantics of
/// `nd_fmm_ref::p2p::p2p` (crate documentation, "Semantics"). The kernel is
/// stateless: it is `Copy`, `Send` and `Sync`, and one value can serve any number of
/// threads.
///
/// In this version every ISA runs the scalar path; the NEON and AVX2 vector kernels
/// follow in Phase 3S T5.
///
/// ```
/// use nd_fmm_simd::{Isa, P2pKernel};
///
/// let kernel = P2pKernel::<f64>::new(Isa::Scalar).unwrap();
/// let sources = [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]];
/// let charges = [2.0, -1.0];
/// let mut potential = [0.0; 2];
/// let mut gradient = [[0.0; 3]; 2];
/// // Targets equal to the sources: the pairs with r² = 0 are skipped.
/// kernel.evaluate(&sources, &charges, &sources, &mut potential, Some(&mut gradient));
/// assert_eq!(potential, [-1.0 / 5.0, 2.0 / 5.0]);
/// assert_eq!(gradient[1], [-2.0 * 3.0 / 125.0, -2.0 * 4.0 / 125.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct P2pKernel<T: SimdScalar> {
    /// The ISA, checked to be available by every constructor.
    isa: Isa,
    /// The precision.
    precision: PhantomData<T>,
}

impl<T: SimdScalar> P2pKernel<T> {
    /// The kernel on `isa`.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] if this machine cannot run `isa` ([`Isa::is_available`]).
    pub fn new(isa: Isa) -> Result<Self, IsaUnavailable> {
        if isa.is_available() {
            Ok(Self {
                isa,
                precision: PhantomData,
            })
        } else {
            Err(IsaUnavailable { isa })
        }
    }

    /// The kernel on the widest available ISA, [`Isa::detect`].
    pub fn detect() -> Self {
        Self {
            isa: Isa::detect(),
            precision: PhantomData,
        }
    }

    /// The ISA the kernel runs on.
    pub fn isa(&self) -> Isa {
        self.isa
    }

    /// P2P: adds the potential of point charges, and optionally its gradient, at each
    /// target (CONVENTIONS §3.1), as `nd_fmm_ref::p2p::p2p` does.
    ///
    /// For each target xᵢ and each source yⱼ with r² = |xᵢ − yⱼ|² ≠ 0,
    ///
    /// φᵢ ← φᵢ + qⱼ / |xᵢ − yⱼ|,  ∇φᵢ ← ∇φᵢ − qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³,
    ///
    /// without the factor 1/(4π). Each target adds its sources in input order,
    /// starting from the value already in `potential` and `gradient`; `gradient`, if
    /// given, holds ∂x, ∂y, ∂z per target. A pair with r² = 0 contributes nothing
    /// (CONVENTIONS §3.13, "Fast kernels"). The kernel neither allocates nor spawns
    /// threads, and writes only into `potential` and `gradient`.
    ///
    /// # Panics
    ///
    /// If `sources.len() != charges.len()`, `potential.len() != targets.len()` or a
    /// given `gradient` has a length other than `targets.len()`, with the messages of
    /// the reference.
    pub fn evaluate(
        &self,
        sources: &[[T; 3]],
        charges: &[T],
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        check_pair(sources.len(), charges.len(), "sources", "charges");
        check_pair(targets.len(), potential.len(), "targets", "potential");
        if let Some(gradient) = gradient.as_deref() {
            check_pair(targets.len(), gradient.len(), "targets", "gradient");
        }
        match self.isa {
            Isa::Scalar => arch::scalar::p2p(sources, charges, targets, potential, gradient),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => arch::neon::p2p(sources, charges, targets, potential, gradient),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => arch::avx2::p2p(sources, charges, targets, potential, gradient),
            // The ISAs of other architectures, which no constructor admits.
            _ => unreachable!("`{}` is not available on this machine", self.isa),
        }
    }
}

/// Panics unless the paired slices `a` and `b` have the same length, with the message
/// of `nd_fmm_ref::p2p`.
fn check_pair(a: usize, b: usize, name_a: &str, name_b: &str) {
    assert_eq!(
        a, b,
        "`{name_a}` and `{name_b}` must have the same length, got {a} and {b}"
    );
}

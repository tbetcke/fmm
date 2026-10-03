//! The P2P kernel bound to one ISA (docs/design/simd-p2p.md §5.1 and §5.2).

use std::marker::PhantomData;

use crate::arch::AvailableIsa;
use crate::{Isa, IsaUnavailable, SimdScalar};

/// A P2P kernel bound to one available [`Isa`].
///
/// [`evaluate`](Self::evaluate) has the signature and semantics of
/// `nd_fmm_ref::p2p::p2p` (crate documentation, "Semantics"). The kernel is
/// stateless: it is `Copy`, `Send` and `Sync`, and one value can serve any number of
/// threads.
///
/// On [`Isa::Neon`] and [`Isa::Avx2`] it runs the vector kernel (crate documentation,
/// "The kernel"), on [`Isa::Scalar`] the reference's loop.
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
    isa: AvailableIsa,
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
        Ok(Self {
            isa: AvailableIsa::new(isa)?,
            precision: PhantomData,
        })
    }

    /// The kernel on the widest available ISA, [`Isa::detect`].
    pub fn detect() -> Self {
        Self::new(Isa::detect()).expect("`Isa::detect` returns an available ISA")
    }

    /// The ISA the kernel runs on.
    pub fn isa(&self) -> Isa {
        self.isa.isa()
    }

    /// P2P: adds the potential of point charges, and optionally its gradient, at each
    /// target (CONVENTIONS §3.1), as `nd_fmm_ref::p2p::p2p` does.
    ///
    /// For each target xᵢ and each source yⱼ with r² = |xᵢ − yⱼ|² ≠ 0,
    ///
    /// φᵢ ← φᵢ + qⱼ / |xᵢ − yⱼ|,  ∇φᵢ ← ∇φᵢ − qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³,
    ///
    /// without the factor 1/(4π); `gradient`, if given, holds ∂x, ∂y, ∂z per target.
    /// The kernel neither allocates nor spawns threads, and writes only into
    /// `potential` and `gradient`.
    ///
    /// - **Coincident pairs.** A pair with r² = 0 contributes nothing (CONVENTIONS
    ///   §3.13, "Fast kernels"); the reference skips xᵢ == yⱼ instead. The rules agree
    ///   on leaf-scaled data. On the vector paths a skipped pair adds a zero, which can
    ///   turn an output of −0 into +0.
    /// - **Order.** Each target adds its sources in input order, starting from the value
    ///   already in `potential` and `gradient`. So evaluating sources `[..k]` and then
    ///   `[k..]` gives the bits of evaluating all of them at once, and a target's result
    ///   does not depend on its position in `targets` or on how many targets there are.
    /// - **Accuracy.** For pairs in the kernel domain, r² = 0 or 2⁻¹⁰⁸ ≤ r² ≤ 2⁷
    ///   (CONVENTIONS §3.13), each potential term is within 8 u_T of the reference's
    ///   term, relative, and each gradient component within 16 u_T relative to
    ///   |qⱼ| / r², with u_T = 2⁻²⁴ (f32) or 2⁻⁵³ (f64). In f32 the gradient contract
    ///   holds for 2⁻⁸⁴ ≤ r² ≤ 2⁷ and |qⱼ| ≤ 1. Outside the domain the result is
    ///   unspecified. The terms are added without compensation, like the reference's.
    /// - **ISAs.** The results are deterministic for one machine, ISA and build, and
    ///   differ between ISAs, and from the reference, within the accuracy above; on
    ///   [`Isa::Scalar`] they equal the reference's bit for bit wherever r² ≠ 0.
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
        T::p2p(self.isa, sources, charges, targets, potential, gradient);
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

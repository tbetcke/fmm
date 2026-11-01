//! The P2P kernel bound to one ISA (docs/design/simd-p2p.md §5.1 and §5.2).

use std::marker::PhantomData;

use nd_fmm_math::Complex;

use crate::arch::AvailableIsa;
use crate::{Isa, IsaUnavailable, SimdScalar};

/// A P2P kernel bound to one available [`Isa`].
///
/// [`evaluate`](Self::evaluate) has the signature and semantics of
/// `nd_fmm_ref::p2p::p2p` (crate documentation, "Semantics"). The kernel is
/// stateless: it is `Copy`, `Send` and `Sync`, and one value can serve any number of
/// threads.
///
/// On [`Isa::Neon`], [`Isa::Avx2`] and [`Isa::Avx512`] it runs the vector kernel (crate documentation,
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

    /// The kernel on the default ISA of this machine, [`Isa::detect`].
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

    /// P2P of `vectors` charge vectors at the same sources and targets (Phase 6 T5, C6.3;
    /// docs/design/optimisation.md §4.3): `charges` holds the m = `vectors` charges of each
    /// source one after the other (`charges[j m + v]`, CONVENTIONS §3.13, "Several charge
    /// vectors"), and `potential` (and `gradient`, if given) the m vectors' values one
    /// vector after the other, m n for n targets (`potential[v n + i]`).
    ///
    /// Vector v of the result equals [`evaluate`](Self::evaluate) of the charges of vector
    /// v into its slices bit for bit, on every ISA: each (target, source) pair forms the
    /// distance and the inverse square root once and then gives every vector exactly the
    /// steps `evaluate` gives it, in source order (crate documentation, "Several charge
    /// vectors"). Every guarantee of `evaluate` therefore holds per vector. The vector
    /// kernels hold a group of vectors in registers (8 without gradients and 4 with, on NEON,
    /// AVX2 and AVX-512), so the distance work is shared within a group; one vector runs
    /// `evaluate` itself.
    ///
    /// # Panics
    ///
    /// If `vectors` is zero, `charges.len() != vectors · sources.len()`,
    /// `potential.len() != vectors · targets.len()` or a given `gradient` has a length
    /// other than `vectors · targets.len()`.
    pub fn evaluate_many(
        &self,
        sources: &[[T; 3]],
        charges: &[T],
        vectors: usize,
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        assert!(vectors > 0, "`vectors` must be at least 1");
        check_pair(
            vectors * sources.len(),
            charges.len(),
            "sources × vectors",
            "charges",
        );
        check_pair(
            vectors * targets.len(),
            potential.len(),
            "targets × vectors",
            "potential",
        );
        if let Some(gradient) = gradient.as_deref() {
            check_pair(
                vectors * targets.len(),
                gradient.len(),
                "targets × vectors",
                "gradient",
            );
        }
        if vectors == 1 {
            T::p2p(self.isa, sources, charges, targets, potential, gradient);
        } else {
            T::p2p_many(
                self.isa, sources, charges, vectors, targets, potential, gradient,
            );
        }
    }

    /// Dipole P2P (Phase 6 T8, C6.4): adds the potential of point dipoles, and optionally
    /// its gradient, at each target (CONVENTIONS §3.7, "Dipole sources"), as
    /// `nd_fmm_ref::p2p::p2p_dipoles` does.
    ///
    /// For each target xᵢ and each source yⱼ with moment dⱼ and r² = |xᵢ − yⱼ|² ≠ 0, with
    /// w = xᵢ − yⱼ,
    ///
    /// φᵢ ← φᵢ + dⱼ · w / r³,  ∇φᵢ ← ∇φᵢ + dⱼ / r³ − 3 (dⱼ · w) w / r⁵,
    ///
    /// without the factor 1/(4π). The guarantees of [`evaluate`](Self::evaluate) hold
    /// with these terms: coincident pairs (r² = 0) contribute nothing, each target adds
    /// its sources in input order (chunk and target-position invariance), the results are
    /// deterministic per machine, ISA and build, and on [`Isa::Scalar`] equal the
    /// reference's bit for bit wherever r² ≠ 0. The vector kernels share the inverse
    /// square root of `evaluate` (crate documentation, "Dipoles").
    ///
    /// - **Accuracy.** For pairs in the kernel domain (CONVENTIONS §3.13), each potential
    ///   term is within 32 u_T of the reference's term relative to |dⱼ| / r², and each
    ///   gradient component within 176 u_T relative to |dⱼ| / r³ (docs/design/simd-p2p.md
    ///   §4.8 derives both). In f32 both hold for 2^(−80 + 2e/3) ≤ r² ≤ 2⁷ with
    ///   |dⱼ,ₖ| ≤ 2^e, e ≥ 0 (CONVENTIONS §3.13, "Range of the terms in f32").
    ///
    /// # Panics
    ///
    /// If `sources.len() != dipoles.len()`, `potential.len() != targets.len()` or a
    /// given `gradient` has a length other than `targets.len()`, with the messages of
    /// the reference.
    pub fn evaluate_dipoles(
        &self,
        sources: &[[T; 3]],
        dipoles: &[[T; 3]],
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        check_pair(sources.len(), dipoles.len(), "sources", "dipoles");
        check_pair(targets.len(), potential.len(), "targets", "potential");
        if let Some(gradient) = gradient.as_deref() {
            check_pair(targets.len(), gradient.len(), "targets", "gradient");
        }
        T::p2p_dipoles(self.isa, sources, dipoles, targets, potential, gradient);
    }

    /// Yukawa P2P (Helmholtz CH2.4b): adds the potential e^{−κr}/r of real point charges,
    /// and optionally its gradient, at each target (CONVENTIONS §3.15, k = iκ), as
    /// `nd_fmm_ref::helmholtz::p2p::p2p_yukawa` does.
    ///
    /// For each target xᵢ and each source yⱼ with r² = |xᵢ − yⱼ|² ≠ 0, r = √r²,
    ///
    /// φᵢ ← φᵢ + qⱼ e^{−κr} / r,  ∇φᵢ ← ∇φᵢ − qⱼ (κr + 1) e^{−κr} (xᵢ − yⱼ) / r³,
    ///
    /// without the factor 1/(4π). The guarantees of [`evaluate`](Self::evaluate) hold
    /// with these terms: coincident pairs (r² = 0) contribute nothing, each target adds
    /// its sources in input order (chunk and target-position invariance), the results are
    /// deterministic per machine, ISA and build, and on [`Isa::Scalar`] equal the
    /// reference's bit for bit wherever r² ≠ 0. The vector kernels run Laplace's loop
    /// with another pair arithmetic (crate documentation, "Yukawa"); at κ = 0 they give
    /// [`evaluate`](Self::evaluate)'s results bit for bit.
    ///
    /// - **Accuracy.** For pairs in the kernel domain (CONVENTIONS §3.13) with
    ///   0 ≤ κr ≤ 64 (f32) or 512 (f64), each potential term is within (20 + 12 κr) u_T of
    ///   the reference's term relative to |qⱼ| e^{−κr} / r, and each gradient component
    ///   within (52 + 12 κr) u_T relative to |qⱼ| (1 + κr) e^{−κr} / r²
    ///   (docs/design/simd-p2p.md §4.10 derives both; the κr part is the conditioning of
    ///   e^{−κr} in r, which the reference shares), for 2⁻²⁰ ≤ |qⱼ| ≤ 1, where the term's
    ///   products stay normal. In f32 the gradient contract holds for 2⁻⁸⁴ ≤ r² ≤ 2⁷, as
    ///   [`evaluate`](Self::evaluate)'s. From κr = −(e_min + 1) ln 2 on (707.7 in f64, 86.6
    ///   in f32) the factor e^{−κr} is replaced by its value there, at most 2^(e_min + 2):
    ///   the potential term is then at most 2^(e_min + 2) |qⱼ| / r, each gradient component
    ///   at most (1 + κr) 2^(e_min + 2) |qⱼ| / r², which grows with κr (and is infinite
    ///   once κr overflows, as the reference's is NaN).
    ///
    /// # Panics
    ///
    /// If κ is not finite or is negative, `sources.len() != charges.len()`,
    /// `potential.len() != targets.len()` or a given `gradient` has a length other than
    /// `targets.len()`, with the messages of the reference for the lengths.
    pub fn evaluate_yukawa(
        &self,
        kappa: T,
        sources: &[[T; 3]],
        charges: &[T],
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        assert!(
            kappa.is_finite() && kappa >= T::zero(),
            "Yukawa P2P needs a finite wavenumber κ ≥ 0, got {}",
            kappa.to_f64()
        );
        check_pair(sources.len(), charges.len(), "sources", "charges");
        check_pair(targets.len(), potential.len(), "targets", "potential");
        if let Some(gradient) = gradient.as_deref() {
            check_pair(targets.len(), gradient.len(), "targets", "gradient");
        }
        T::p2p_yukawa(
            self.isa, kappa, sources, charges, targets, potential, gradient,
        );
    }

    /// Helmholtz P2P (Helmholtz CH3.4b): adds the potential e^{ikr}/r of real or complex
    /// point charges, and optionally its gradient, at each target (CONVENTIONS §3.15), as
    /// `nd_fmm_ref::helmholtz::p2p::p2p` does, with the outputs in split planes.
    ///
    /// For each target xᵢ and each source yⱼ with r² = |xᵢ − yⱼ|² ≠ 0, r = √r²,
    ///
    /// φᵢ ← φᵢ + qⱼ e^{ikr} / r,  ∇φᵢ ← ∇φᵢ + qⱼ (ikr − 1) e^{ikr} (xᵢ − yⱼ) / r³,
    ///
    /// without the factor 1/(4π), for a wavenumber k with Re k ≥ 0 and Im k ≥ 0 in the
    /// coordinates' units (in the FMM the leaf-scaled ones of CONVENTIONS §3.13). The
    /// charges are [`HelmholtzCharges`]; `potential` holds 2n values for n targets, the real
    /// plane and then the imaginary one (`potential[i]` = Re φᵢ, `potential[n + i]` = Im φᵢ),
    /// and `gradient`, if given, likewise 2n gradients: the layout of
    /// [`evaluate_many`](Self::evaluate_many)'s two output vectors, and of a complex vector
    /// in CONVENTIONS §3.15's leaf chunks.
    ///
    /// The guarantees of [`evaluate`](Self::evaluate) hold with these terms: coincident
    /// pairs (r² = 0) contribute nothing, each target adds its sources in input order (chunk
    /// and target-position invariance), the results are deterministic per machine, ISA and
    /// build, and on [`Isa::Scalar`] equal the reference's bit for bit wherever r² ≠ 0 (a
    /// real charge taken as the complex one with Im q = 0). The vector kernels run Laplace's
    /// loop with another pair arithmetic and two output planes (crate documentation,
    /// "Helmholtz"); with real charges their real plane at k = 0 is
    /// [`evaluate`](Self::evaluate)'s and at k = iκ [`evaluate_yukawa`](Self::evaluate_yukawa)'s
    /// bit for bit. The choice between their entry points (charges, Im k > 0, gradients) is
    /// made once per call.
    ///
    /// - **Accuracy.** For pairs in the kernel domain (CONVENTIONS §3.13) with
    ///   Re(k) r ≤ 2¹⁰ and Im(k) r ≤ 64 (f32) or 512 (f64), each part of each potential term
    ///   is within (32 + 12 |k| r) u_T of the reference's term relative to
    ///   |qⱼ| e^{−Im(k) r} / r, and each part of each gradient component within
    ///   (66 + 12 |k| r) u_T relative to |qⱼ| (1 + |k| r) e^{−Im(k) r} / r²
    ///   (docs/design/simd-p2p.md §4.12 derives both; the |k| r part is the conditioning of
    ///   e^{ikr} in r, which the reference shares), for 2⁻²⁰ ≤ |qⱼ| ≤ 1. In f32 the gradient
    ///   contract holds for (1 + |k| r) / r³ ≤ 2¹²⁶ and r² ≤ 2⁷, [`evaluate`](Self::evaluate)'s
    ///   range 2⁻⁸⁴ ≤ r² at k = 0. Beyond Re(k) r = 2¹⁰ the terms are unspecified (the sine
    ///   and cosine's domain, [`crate::sincos`]); the FMM's leaf rule keeps Re(k) r below
    ///   22 c_leaf. Beyond the Im(k) r limits the damping behaves as
    ///   [`evaluate_yukawa`](Self::evaluate_yukawa)'s.
    ///
    /// # Panics
    ///
    /// If Re k or Im k is not finite or is negative; if the charges do not hold one value
    /// (real) or two (complex) per source; or if `potential`, or a given `gradient`, does
    /// not hold two values per target.
    pub fn evaluate_helmholtz(
        &self,
        k: Complex<T>,
        sources: &[[T; 3]],
        charges: HelmholtzCharges<'_, T>,
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        assert!(
            k.re.is_finite() && k.im.is_finite() && k.re >= T::zero() && k.im >= T::zero(),
            "Helmholtz P2P needs a finite wavenumber k with Re k ≥ 0 and Im k ≥ 0, got {} + {}i",
            k.re.to_f64(),
            k.im.to_f64()
        );
        let (charges, complex) = match charges {
            HelmholtzCharges::Real(q) => {
                check_pair(sources.len(), q.len(), "sources", "charges");
                (q, false)
            }
            HelmholtzCharges::Complex(q) => {
                check_pair(2 * sources.len(), q.len(), "sources × 2", "charges");
                (q, true)
            }
        };
        check_pair(
            2 * targets.len(),
            potential.len(),
            "targets × 2",
            "potential",
        );
        if let Some(gradient) = gradient.as_deref() {
            check_pair(2 * targets.len(), gradient.len(), "targets × 2", "gradient");
        }
        T::p2p_helmholtz(
            self.isa, k.re, k.im, complex, sources, charges, targets, potential, gradient,
        );
    }
}

/// The charges of [`P2pKernel::evaluate_helmholtz`] (Helmholtz CH3.4b).
#[derive(Clone, Copy, Debug)]
pub enum HelmholtzCharges<'a, T> {
    /// Real charges, one value per source.
    Real(&'a [T]),
    /// Complex charges, Re q and Im q of each source one after the other (point-major:
    /// `charges[2 j]` and `charges[2 j + 1]`).
    Complex(&'a [T]),
}

/// Panics unless the paired slices `a` and `b` have the same length, with the message
/// of `nd_fmm_ref::p2p`.
fn check_pair(a: usize, b: usize, name_a: &str, name_b: &str) {
    assert_eq!(
        a, b,
        "`{name_a}` and `{name_b}` must have the same length, got {a} and {b}"
    );
}

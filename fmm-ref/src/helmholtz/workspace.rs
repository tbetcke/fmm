//! Caller-owned scratch memory for the Helmholtz operators.

use nd_fmm_math::orthonormal::DerivativeScratch;
use nd_fmm_math::{Complex, Layout, RealScalar};

/// Scratch memory for the Helmholtz operators of [`helmholtz`](crate::helmholtz), built
/// once for a maximum degree, as Laplace's [`Workspace`](crate::Workspace).
///
/// A workspace built for degree `p` serves every call with degree at most `p`; an operator
/// called with a larger degree panics. Its contents between calls are unspecified. It
/// holds the real orthonormal harmonics of a point and their two angular derivatives
/// ((p + 1)² reals each, CONVENTIONS §3.6), the scratch of
/// [`harmonics_derivatives`](nd_fmm_math::orthonormal::harmonics_derivatives), and the
/// radial functions of degrees 0 to p with their derivatives, complex (the complex path)
/// and real (Yukawa's real path).
///
/// ```
/// use nd_fmm_ref::helmholtz::Workspace;
///
/// let ws = Workspace::<f64>::new(8);
/// assert_eq!(ws.p(), 8);
/// ```
pub struct Workspace<T: RealScalar> {
    p: usize,
    harmonics: Vec<T>,
    d_theta: Vec<T>,
    d_phi: Vec<T>,
    scratch: DerivativeScratch<T>,
    radial: Vec<Complex<T>>,
    radial_derivative: Vec<Complex<T>>,
    real: Vec<T>,
    real_derivative: Vec<T>,
}

/// The buffers of one operator call of degree p, borrowed from a [`Workspace`].
pub(crate) struct Buffers<'a, T: RealScalar> {
    /// Yₙᵐ, (p + 1)².
    pub harmonics: &'a mut [T],
    /// ∂θ Yₙᵐ, (p + 1)².
    pub d_theta: &'a mut [T],
    /// (1/sin θ) ∂φ Yₙᵐ, (p + 1)².
    pub d_phi: &'a mut [T],
    /// The scratch of `harmonics_derivatives`.
    pub scratch: &'a mut DerivativeScratch<T>,
    /// A complex radial function of degrees 0 to p.
    pub radial: &'a mut [Complex<T>],
    /// Its derivative.
    pub radial_derivative: &'a mut [Complex<T>],
    /// A real radial function of degrees 0 to p.
    pub real: &'a mut [T],
    /// Its derivative.
    pub real_derivative: &'a mut [T],
}

impl<T: RealScalar> Workspace<T> {
    /// Allocates a workspace for operators of degree at most `p`.
    pub fn new(p: usize) -> Self {
        let len = Layout::new(p).len();
        let zero = Complex::new(T::zero(), T::zero());
        Self {
            p,
            harmonics: vec![T::zero(); len],
            d_theta: vec![T::zero(); len],
            d_phi: vec![T::zero(); len],
            scratch: DerivativeScratch::new(p),
            radial: vec![zero; p + 1],
            radial_derivative: vec![zero; p + 1],
            real: vec![T::zero(); p + 1],
            real_derivative: vec![T::zero(); p + 1],
        }
    }

    /// Largest degree p this workspace serves.
    pub fn p(&self) -> usize {
        self.p
    }

    /// The buffers for degree `p`.
    ///
    /// # Panics
    ///
    /// If `p` exceeds [`p`](Self::p).
    pub(crate) fn buffers(&mut self, p: usize) -> Buffers<'_, T> {
        assert!(
            p <= self.p,
            "Workspace built for p = {} is too small for p = {p}",
            self.p
        );
        let len = Layout::new(p).len();
        Buffers {
            harmonics: &mut self.harmonics[..len],
            d_theta: &mut self.d_theta[..len],
            d_phi: &mut self.d_phi[..len],
            scratch: &mut self.scratch,
            radial: &mut self.radial[..=p],
            radial_derivative: &mut self.radial_derivative[..=p],
            real: &mut self.real[..=p],
            real_derivative: &mut self.real_derivative[..=p],
        }
    }
}
